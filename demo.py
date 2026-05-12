"""
GluLLM Synthetic Data Demo
Train and test GluLLM with synthetic glucose data
No real data required - generates realistic synthetic glucose time series
"""
import datetime
import pickle
import numpy as np
import torch
import torch.nn as nn
import os 
# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from models import GluLLM, TDI
from args_generator import args_bglp


class TDIEmbedder:
    """
    Time-Dependent Information (TDI) embedder.
    Produces embeddings of shape [n_windows, embed_dim] per call,
    which the dataset assembles into [n_patches, embed_dim] per sample.

    Mirrors __llm_embedding_event__ logic:
      prompt = "Historical data from {start} to {end}. {bolus:.1f} units of bolus insulin delivered."
    """

    def __init__(self, cofigs, token_len: int, batch_size: int = 200):
        """
        Args:
            llm_model : LLM/sentence-transformer that accepts List[str]
                        and returns Tensor of shape (B, embed_dim).
            token_len : Number of 5-min timesteps per patch/token window.
            batch_size: Max prompts per LLM forward pass.
        """
        self.model = TDI.Model(cofigs)
        self.token_len  = token_len
        self.batch_size = batch_size

    def embed(
        self,
        start_indices: np.ndarray,
        base_datetime: datetime.datetime,
        bolus_series: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute TDI embeddings for a list of patch start indices.

        Args:
            start_indices : 1-D int array of shape (N,) — patch start positions
                            in 5-min steps from base_datetime.
            base_datetime : datetime at index 0 of the glucose series.
            bolus_series  : bolus values aligned with glucose series, shape (T,).

        Returns:
            Tensor of shape (N, embed_dim)
        """
        output_list = []
        n_windows   = len(start_indices)
        num_batches = (n_windows + self.batch_size - 1) // self.batch_size

        for nb in range(num_batches):
            prompt_list = []
            batch_start = nb * self.batch_size
            batch_end   = min((nb + 1) * self.batch_size, n_windows)

            for ind in range(batch_start, batch_end):
                patch_start_idx = int(start_indices[ind])

                # Time span of this patch
                start_dt = base_datetime + datetime.timedelta(
                    minutes=5 * patch_start_idx
                )
                end_dt = base_datetime + datetime.timedelta(
                    minutes=5 * (patch_start_idx + self.token_len - 1)
                )
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                end_str   = end_dt.strftime("%Y-%m-%d %H:%M:%S")

                # Bolus sum over this patch
                bolus_window = bolus_series[
                    patch_start_idx : patch_start_idx + self.token_len
                ]
                total_bolus = float(np.sum(bolus_window))

                # Prompt mirrors __llm_embedding_event__
                prompt = (
                    f"Historical data from {start_str} to {end_str}. "
                    f"{total_bolus:.1f} units of bolus insulin delivered."
                )
                prompt_list.append(prompt)

            # LLM forward pass -> (len(prompt_list), embed_dim)
            output = self.model(prompt_list)
            output_list.append(output.detach().cpu())

            del prompt_list, output
            torch.cuda.empty_cache()

        return torch.cat(output_list, dim=0)  # (N, embed_dim)


class SyntheticGlucoseDataset(Dataset):
    """
    Synthetic glucose dataset.

    x_mark / y_mark are LLM-based TDI embeddings with shape:
        [batch*vars, n_patches, embed_dim]
    which in a univariate setting (vars=1) becomes:
        [batch, n_patches, embed_dim]

    Each sample returns:
        x_mark : (n_x_patches, embed_dim)  covers input window patches
        y_mark : (n_y_patches, embed_dim)  covers prediction window patches
    """

    BASE_DATETIME = datetime.datetime(2024, 1, 1, 0, 0, 0)

    def __init__(
        self,
        n_subjects   : int,
        seq_len      : int,
        pred_len     : int,
        token_len    : int,
        tdi_embedder : TDIEmbedder,
        scaler       = None,
        split        : str = "train",
    ):
        assert seq_len  % token_len == 0, "seq_len must be divisible by token_len"
        assert pred_len % token_len == 0, "pred_len must be divisible by token_len"

        self.n_subjects      = n_subjects
        self.seq_len         = seq_len
        self.pred_len        = pred_len
        self.token_len       = token_len
        self.tdi_embedder    = tdi_embedder
        self.split           = split

        # Number of patches per window
        self.n_x_patches = seq_len  // token_len
        self.n_y_patches = pred_len // token_len

        np.random.seed(
            2024 if split == "train" else
            2025 if split == "val"   else
            2026
        )

        self.data = self._generate_glucose_data()

        # Scaler
        if scaler is None:
            all_glucose = np.concatenate([d["glucose"] for d in self.data])
            self.mean   = float(np.mean(all_glucose))
            self.std    = float(np.std(all_glucose))
        else:
            self.mean = scaler["mean"]
            self.std  = scaler["std"]

        for subject_data in self.data:
            subject_data["glucose_norm"] = (
                (subject_data["glucose"] - self.mean) / self.std
            )

        self.samples_per_subject = (
            len(self.data[0]["glucose"]) - seq_len - pred_len + 1
        )

        # Pre-compute TDI embeddings
        self._precompute_tdi_embeddings()

    def _generate_glucose_data(self):
        """Generate realistic synthetic glucose + bolus time series."""
        data = []
        for subject_id in range(self.n_subjects):
            n_tp  = 7 * 24 * 12  # 7 days of 5-min readings
            base  = np.random.uniform(140, 180)
            hours = np.arange(n_tp) / 12

            # Diurnal pattern
            diurnal = 20 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)

            # Meal effects
            meal_effect = np.zeros(n_tp)
            for day in range(7):
                for meal_hour in [7, 12, 18]:
                    meal_time = (day * 24 + meal_hour) * 12
                    if meal_time < n_tp:
                        spike = np.random.uniform(40, 80)
                        td    = (np.arange(n_tp) - meal_time) / 12
                        mask  = (td >= 0) & (td <= 3)
                        meal_effect[mask] += spike * np.exp(
                            -((td[mask] - 0.5) ** 2) / 0.5
                        )

            # Random walk
            rw  = np.cumsum(np.random.normal(0, 2, n_tp))
            rw -= rw.mean()

            # Combine + noise + clip
            glucose = np.clip(
                base + diurnal + meal_effect + rw
                + np.random.normal(0, 5, n_tp),
                40, 400,
            )

            # Synthetic bolus ~30 min after each meal
            bolus = np.zeros(n_tp)
            for day in range(7):
                for meal_hour in [7, 12, 18]:
                    bolus_time = (day * 24 + meal_hour) * 12 + 6
                    if bolus_time < n_tp:
                        bolus[bolus_time] = np.random.uniform(2.0, 8.0)

            age    = np.random.randint(20, 70)
            gender = np.random.choice(["M", "F"])
            bmi    = np.random.uniform(22, 35)
            hba1c  = np.random.uniform(6.0, 9.5)

            data.append({
                "subject_id"   : f"SYNTH_{self.split}_{subject_id:04d}",
                "glucose"      : glucose,
                "bolus"        : bolus,
                "prompt"       : (
                    f"Characteristics: - Age: {age}, - Gender: {gender}, "
                    f"- BMI: {bmi:.1f}, - HbA1c: {hba1c:.1f}, "
                    f"Predict next value based on historical glucose embedding: "
                ),
                "demographics" : {
                    "age": age, "gender": gender,
                    "bmi": bmi, "hba1c": hba1c,
                },
            })
        return data

    def _precompute_tdi_embeddings(self):
        """
        Pre-compute TDI embeddings for every (subject, window, patch).

        For each subject we collect ALL patch start indices across all
        sliding windows in a single batched LLM call, then reshape into
        [n_windows, n_patches, embed_dim].

        Stored as:
            self.x_tdi[subject_idx]  Tensor (n_windows, n_x_patches, embed_dim)
            self.y_tdi[subject_idx]  Tensor (n_windows, n_y_patches, embed_dim)
        """
        print(f"\nPre-computing TDI embeddings [{self.split}] — "
              f"{self.n_subjects} subjects, "
              f"n_x_patches={self.n_x_patches}, "
              f"n_y_patches={self.n_y_patches}")

        self.x_tdi = []
        self.y_tdi = []

        for subject_data in tqdm(self.data, desc=f"TDI [{self.split}]"):
            bolus     = subject_data["bolus"]
            n_windows = self.samples_per_subject

            # x patches
            # Window w starts at index w; patch p starts at w + p * token_len
            window_indices  = np.arange(n_windows)                           # (n_windows,)
            x_patch_offsets = np.arange(self.n_x_patches) * self.token_len   # (n_x_patches,)

            # Broadcast -> (n_windows, n_x_patches) -> flatten
            x_patch_starts = (
                window_indices[:, None] + x_patch_offsets[None, :]
            ).reshape(-1)  # (n_windows * n_x_patches,)

            x_embeds_flat = self.tdi_embedder.embed(
                x_patch_starts, self.BASE_DATETIME, bolus
            )  # (n_windows * n_x_patches, embed_dim)

            embed_dim     = x_embeds_flat.shape[-1]
            x_tdi_subject = x_embeds_flat.reshape(
                n_windows, self.n_x_patches, embed_dim
            )  # (n_windows, n_x_patches, embed_dim)

            # y patches
            # y window w starts at index w + seq_len
            y_base          = window_indices + self.seq_len                   # (n_windows,)
            y_patch_offsets = np.arange(self.n_y_patches) * self.token_len   # (n_y_patches,)

            y_patch_starts = (
                y_base[:, None] + y_patch_offsets[None, :]
            ).reshape(-1)  # (n_windows * n_y_patches,)

            y_embeds_flat = self.tdi_embedder.embed(
                y_patch_starts, self.BASE_DATETIME, bolus
            )  # (n_windows * n_y_patches, embed_dim)

            y_tdi_subject = y_embeds_flat.reshape(
                n_windows, self.n_y_patches, embed_dim
            )  # (n_windows, n_y_patches, embed_dim)

            self.x_tdi.append(x_tdi_subject)
            self.y_tdi.append(y_tdi_subject)

    def __len__(self):
        return self.n_subjects * self.samples_per_subject

    def __getitem__(self, idx):
        subject_idx = idx // self.samples_per_subject
        window_idx  = idx %  self.samples_per_subject

        subject_data = self.data[subject_idx]
        start_idx    = window_idx
        end_idx      = start_idx + self.seq_len
        pred_end_idx = end_idx   + self.pred_len

        # Glucose tensors
        x = torch.FloatTensor(
            subject_data["glucose_norm"][start_idx:end_idx].reshape(-1, 1)
        )  # (seq_len, 1)
        y = torch.FloatTensor(
            subject_data["glucose_norm"][end_idx:pred_end_idx].reshape(-1, 1)
        )  # (pred_len, 1)

        # TDI embeddings
        # x_mark : (n_x_patches, embed_dim)
        # y_mark : (n_y_patches, embed_dim)
        # After DataLoader collation:
        #   x_mark -> (B, n_x_patches, embed_dim) = [batch*vars, n_patches, embed_dim]
        #   y_mark -> (B, n_y_patches, embed_dim) = [batch*vars, n_patches, embed_dim]
        x_mark = self.x_tdi[subject_idx][window_idx].float()
        y_mark = self.y_tdi[subject_idx][window_idx].float()

        return x, y, x_mark, y_mark, subject_data["prompt"]

    def get_scaler(self):
        return {"mean": self.mean, "std": self.std}


class SyntheticTrainer:
    """Training pipeline with synthetic data."""

    def __init__(self, args):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Device      : {self.device}")
        print(f"✓ Model       : {self.args.mn}")
        print(f"✓ Checkpoint  : {self.checkpoint_dir}")

    def prepare_dataloaders(self, tdi_embedder: TDIEmbedder):
        """Create synthetic data loaders with TDI embeddings."""
        print("\nGenerating synthetic datasets...")

        train_dataset = SyntheticGlucoseDataset(
            n_subjects   = self.args.n_train_subjects,
            seq_len      = self.args.seq_len,
            pred_len     = self.args.test_pred_len,
            token_len    = self.args.token_len,
            tdi_embedder = tdi_embedder,
            split        = "train",
        )

        scaler = train_dataset.get_scaler()

        val_dataset = SyntheticGlucoseDataset(
            n_subjects   = self.args.n_val_subjects,
            seq_len      = self.args.seq_len,
            pred_len     = self.args.test_pred_len,
            token_len    = self.args.token_len,
            tdi_embedder = tdi_embedder,
            scaler       = scaler,
            split        = "val",
        )

        test_dataset = SyntheticGlucoseDataset(
            n_subjects   = self.args.n_test_subjects,
            seq_len      = self.args.seq_len,
            pred_len     = self.args.test_pred_len,
            token_len    = self.args.token_len,
            tdi_embedder = tdi_embedder,
            scaler       = scaler,
            split        = "test",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size  = self.args.batch_size,
            shuffle     = True,
            num_workers = 4,
            pin_memory  = True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size  = self.args.batch_size,
            shuffle     = False,
            num_workers = 4,
            pin_memory  = True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size  = self.args.batch_size,
            shuffle     = False,
            num_workers = 4,
            pin_memory  = True,
        )

        # Save scaler
        scaler_path = self.checkpoint_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        print(f"✓ Train  : {len(train_dataset):,} samples from {self.args.n_train_subjects} subjects")
        print(f"✓ Val    : {len(val_dataset):,} samples from {self.args.n_val_subjects} subjects")
        print(f"✓ Test   : {len(test_dataset):,} samples from {self.args.n_test_subjects} subjects")
        print(f"✓ Scaler : mean={scaler['mean']:.2f}, std={scaler['std']:.2f}")

        return train_loader, val_loader, test_loader, scaler

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train one epoch."""
        model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        for batch_x, batch_y, batch_x_mark, batch_y_mark, prompts in pbar:
            # batch_x_mark : (B, n_x_patches, embed_dim) = [batch*vars, n_patches, embed_dim]
            # batch_y_mark : (B, n_y_patches, embed_dim) = [batch*vars, n_patches, embed_dim]
            batch_x      = batch_x.to(self.device,      dtype=torch.bfloat16)
            batch_y      = batch_y.to(self.device,      dtype=torch.bfloat16)
            batch_x_mark = batch_x_mark.to(self.device, dtype=torch.bfloat16)
            batch_y_mark = batch_y_mark.to(self.device, dtype=torch.bfloat16)

            optimizer.zero_grad()

            outputs = model(batch_x, batch_x_mark, None, batch_y_mark, prompts)
            outputs = outputs[:, -self.args.token_len:, :]
            batch_y = batch_y[:, -self.args.token_len:, :]

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg" : f"{total_loss / n_batches:.4f}",
            })

        return total_loss / n_batches

    def validate(self, model, val_loader, criterion):
        """Validate model."""
        model.eval()
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, prompts in val_loader:
                batch_x      = batch_x.to(self.device,      dtype=torch.bfloat16)
                batch_y      = batch_y.to(self.device,      dtype=torch.bfloat16)
                batch_x_mark = batch_x_mark.to(self.device, dtype=torch.bfloat16)
                batch_y_mark = batch_y_mark.to(self.device, dtype=torch.bfloat16)

                outputs = model(batch_x, batch_x_mark, None, batch_y_mark, prompts)
                outputs = outputs[:, -self.args.token_len:, :]
                batch_y = batch_y[:, -self.args.token_len:, :]

                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                n_batches  += 1

        return total_loss / n_batches

    def train(self, tdi_embedder: TDIEmbedder):
        """Full training pipeline."""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader, scaler = self.prepare_dataloaders(tdi_embedder)

        print("\nInitializing model...")
        model    = GluLLM.Model(self.args).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model: {n_params:,} parameters")

        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.args.train_epochs, eta_min=1e-8
        )

        best_val_loss    = float("inf")
        patience_counter = 0
        best_model_path  = self.checkpoint_dir / f"checkpoint_{self.args.mn}.pth"

        print(f"\n{'='*60}")
        print("Training Progress")
        print(f"{'='*60}")

        for epoch in range(1, self.args.train_epochs + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            val_loss   = self.validate(model, val_loader, criterion)
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{self.args.train_epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.8f}"
            )

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ Saved (Val: {val_loss:.6f})")
            else:
                patience_counter += 1

            if patience_counter >= self.args.patience:
                print(f"  → Early stop at epoch {epoch}")
                break

        print(f"\n{'='*60}")
        print("Training Complete")
        print(f"{'='*60}")
        print(f"✓ Best Val Loss : {best_val_loss:.6f}")
        print(f"✓ Model saved   : {best_model_path}")

        return best_model_path


class SyntheticTester:
    """Testing pipeline with synthetic data."""

    def __init__(self, args, model_path, scaler_path, tdi_embedder: TDIEmbedder):
        self.args         = args
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tdi_embedder = tdi_embedder

        # Load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load model
        self.model = GluLLM.Model(self.args).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        print(f"✓ Model loaded : {model_path}")
        print(f"✓ Scaler       : mean={self.scaler['mean']:.2f}, std={self.scaler['std']:.2f}")

    def test(self):
        """Run testing on synthetic test set."""
        print(f"\n{'='*60}")
        print("Starting Testing")
        print(f"{'='*60}")

        test_dataset = SyntheticGlucoseDataset(
            n_subjects   = self.args.n_test_subjects,
            seq_len      = self.args.seq_len,
            pred_len     = self.args.test_pred_len,
            token_len    = self.args.token_len,
            tdi_embedder = self.tdi_embedder,
            scaler       = self.scaler,
            split        = "test",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size  = self.args.batch_size,
            shuffle     = False,
            num_workers = 4,
        )

        print(f"✓ Test: {len(test_dataset):,} samples from {self.args.n_test_subjects} subjects")

        all_preds = []
        all_trues = []

        print("\nRunning inference...")
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, prompts in tqdm(test_loader):
                # batch_x_mark : (B, n_x_patches, embed_dim) = [batch*vars, n_patches, embed_dim]
                # batch_y_mark : (B, n_y_patches, embed_dim) = [batch*vars, n_patches, embed_dim]
                batch_x      = batch_x.to(self.device,      dtype=torch.bfloat16)
                batch_y      = batch_y.to(self.device,      dtype=torch.bfloat16)
                batch_x_mark = batch_x_mark.to(self.device, dtype=torch.bfloat16)
                batch_y_mark = batch_y_mark.to(self.device, dtype=torch.bfloat16)

                pred_len  = self.args.test_pred_len
                token_len = self.args.token_len
                n_steps   = pred_len // token_len

                step_predictions = []
                current_x        = batch_x        # (B, seq_len, 1)
                current_x_mark   = batch_x_mark   # (B, n_x_patches, embed_dim)

                for step in range(n_steps):
                    # Slice y_mark for this autoregressive step -> (B, 1, embed_dim)
                    step_y_mark = batch_y_mark[:, step:step + 1, :]

                    outputs = self.model(
                        current_x,
                        current_x_mark,
                        None,
                        step_y_mark,
                        prompts,
                    )  # (B, token_len, 1)

                    pred_patch = outputs[:, -token_len:, :]  # (B, token_len, 1)
                    step_predictions.append(pred_patch)

                    if step < n_steps - 1:
                        # Roll glucose input: drop oldest patch, append prediction
                        current_x = torch.cat(
                            [current_x[:, token_len:, :], pred_patch.detach()], dim=1
                        )  # (B, seq_len, 1)

                        # Roll x_mark: drop oldest patch embed, append current y patch embed
                        next_x_patch_mark = batch_y_mark[:, step:step + 1, :]
                        current_x_mark = torch.cat(
                            [current_x_mark[:, 1:, :], next_x_patch_mark], dim=1
                        )  # (B, n_x_patches, embed_dim)

                # Concatenate steps -> (B, pred_len, 1)
                predictions = torch.cat(step_predictions, dim=1)[:, :pred_len, :]

                all_preds.append(predictions.cpu().float().numpy())
                all_trues.append(batch_y[:, :pred_len, :].cpu().float().numpy())

        # Concatenate all batches
        predictions  = np.concatenate(all_preds)   # (N, pred_len, 1)
        ground_truth = np.concatenate(all_trues)   # (N, pred_len, 1)

        # Denormalize
        pred_mg = predictions  * self.scaler["std"] + self.scaler["mean"]
        true_mg = ground_truth * self.scaler["std"] + self.scaler["mean"]

        # Overall metrics
        rmse = np.sqrt(np.mean((pred_mg - true_mg) ** 2))
        mae  = np.mean(np.abs(pred_mg - true_mg))
        mape = np.mean(np.abs((pred_mg - true_mg) / (true_mg + 1e-8))) * 100

        # 30-min (index 5) and 60-min (index 11) specific metrics
        rmse_30 = np.sqrt(np.mean((pred_mg[:, 5, 0] - true_mg[:, 5, 0]) ** 2))
        mae_30  = np.mean(np.abs(pred_mg[:, 5, 0] - true_mg[:, 5, 0]))
        rmse_60 = np.sqrt(np.mean((pred_mg[:, 11, 0] - true_mg[:, 11, 0]) ** 2))
        mae_60  = np.mean(np.abs(pred_mg[:, 11, 0] - true_mg[:, 11, 0]))

        print(f"\n{'='*60}")
        print("Test Results")
        print(f"{'='*60}")
        print(f"Overall:")
        print(f"  RMSE : {rmse:.2f} mg/dL")
        print(f"  MAE  : {mae:.2f} mg/dL")
        print(f"  MAPE : {mape:.2f}%")
        print(f"\n30-minute ahead:")
        print(f"  RMSE : {rmse_30:.2f} mg/dL")
        print(f"  MAE  : {mae_30:.2f} mg/dL")
        print(f"\n60-minute ahead:")
        print(f"  RMSE : {rmse_60:.2f} mg/dL")
        print(f"  MAE  : {mae_60:.2f} mg/dL")

        return {
            "predictions" : pred_mg,
            "ground_truth": true_mg,
            "rmse"        : rmse,
            "mae"         : mae,
            "mape"        : mape,
            "rmse_30"     : rmse_30,
            "mae_30"      : mae_30,
            "rmse_60"     : rmse_60,
            "mae_60"      : mae_60,
        }


def main():
    """Main execution."""
    # Use args_bglp directly — add synthetic-specific fields if missing
    args = args_bglp()

    if not hasattr(args, "mode"):
        args.mode = "train_test"
    if not hasattr(args, "n_train_subjects"):
        args.n_train_subjects = 10
    if not hasattr(args, "n_val_subjects"):
        args.n_val_subjects = 2
    if not hasattr(args, "n_test_subjects"):
        args.n_test_subjects = 3
    if not hasattr(args, "checkpoint_dir"):
        args.checkpoint_dir = "./checkpoints_synthetic"
    if not hasattr(args, "seq_len"):
        args.seq_len = 24
    if not hasattr(args, "token_len"):
        args.token_len = 12
    if not hasattr(args, "test_pred_len"):
        args.test_pred_len = 12
    if not hasattr(args, "seed"):
        args.seed = 2026

    # Validate divisibility
    assert args.seq_len       % args.token_len == 0, "seq_len must be divisible by token_len"
    assert args.test_pred_len % args.token_len == 0, "test_pred_len must be divisible by token_len"

    print(f"\n{'='*60}")
    print("GluLLM Synthetic Data Demo")
    print(f"{'='*60}")
    print(f"Mode            : {args.mode}")
    print(f"Model           : {args.mn}")
    print(f"Train subjects  : {args.n_train_subjects}")
    print(f"Val subjects    : {args.n_val_subjects}")
    print(f"Test subjects   : {args.n_test_subjects}")
    print(f"Epochs          : {args.train_epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"seq_len         : {args.seq_len}")
    print(f"token_len       : {args.token_len}")
    print(f"test_pred_len   : {args.test_pred_len}")
    print(f"n_x_patches     : {args.seq_len       // args.token_len}")
    print(f"n_y_patches     : {args.test_pred_len // args.token_len}")

    model_path  = Path(args.checkpoint_dir) / f"checkpoint_{args.mn}.pth"
    scaler_path = Path(args.checkpoint_dir) / "scaler.pkl"

    # Initialise GluLLM backbone for TDI embedder
    print("\nInitialising LLM backbone for TDI embedder...")
    glullm_backbone = GluLLM.Model(args)

    tdi_embedder = TDIEmbedder(
        args,
        token_len  = args.token_len,
        batch_size = 200,
    )

    # Training
    if args.mode in ["train", "train_test"]:
        trainer    = SyntheticTrainer(args)
        model_path = trainer.train(tdi_embedder)

    # Testing
    if args.mode in ["test", "train_test"]:
        if not model_path.exists():
            print(f"\n❌ Model not found at {model_path}")
            print("Run with args.mode = 'train' first")
            return

        tester  = SyntheticTester(args, model_path, scaler_path, tdi_embedder)
        results = tester.test()

    print(f"\n{'='*60}")
    print("Demo Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()