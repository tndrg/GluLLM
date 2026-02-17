"""
GluLLM: Empowering digital health management with on-device
large language models for glucose prediction

Author: Taiyu Zhu
Affiliation: University of Oxford
Version: 1.0.0
"""

import os
import random
import pickle
import warnings
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import GluLLM
from data_provider.data_factory import data_provider
from utils.metrics import metric, pred_metrics
from args_generator import args_bglp
from sklearn.model_selection import KFold
from tqdm import tqdm


warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GlucosePredictionExperiment:
    """Main experiment class for glucose prediction using GluLLM."""
    
    def __init__(self, args):
        """Initialize experiment with parsed arguments."""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = GluLLM.Model(args).to(self.device)
        
        # Setup directories
        self.checkpoint_dir = Path(args.checkpoint_dir) / args.ds
        self.results_dir = Path(args.results_dir) / args.ds
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = None
        
    def prepare_data(self, split: str) -> Tuple[DataLoader, Optional[object]]:
        """Prepare data loader for given split."""
        flag = f"pop_{split}" if self.args.population else split
        dataset, loader, scaler = data_provider(self.args, flag, self.scaler)
        
        if split == 'train' and self.args.population:
            scaler_path = self.checkpoint_dir / 'scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            self.scaler = scaler
            
        return loader, scaler
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch) -> float:
        """Train for one epoch with progress bar."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        
        # Add progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=True)
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark, prompts in pbar:
            batch_x = batch_x.to(self.device, dtype=torch.bfloat16)
            batch_y = batch_y.to(self.device, dtype=torch.bfloat16)
            batch_x_mark = batch_x_mark.to(self.device, dtype=torch.bfloat16)
            batch_y_mark = batch_y_mark.to(self.device, dtype=torch.bfloat16)
            
            optimizer.zero_grad()
            
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompts)
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :]
                    loss = criterion(outputs, batch_y)
            else:
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompts)
                outputs = outputs[:, -self.args.token_len:, :]
                batch_y = batch_y[:, -self.args.token_len:, :]
                loss = criterion(outputs, batch_y)
            
            if self.args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'avg_loss': f'{total_loss/n_batches:.4f}'})
        
        return total_loss / n_batches
    
    def evaluate(self, data_loader, criterion) -> float:
        """Evaluate model on validation/test set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, prompts in data_loader:
                batch_x = batch_x.to(self.device, dtype=torch.bfloat16)
                batch_y = batch_y.to(self.device, dtype=torch.bfloat16)
                batch_x_mark = batch_x_mark.to(self.device, dtype=torch.bfloat16)
                batch_y_mark = batch_y_mark.to(self.device, dtype=torch.bfloat16)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompts)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompts)
                    
                outputs = outputs[:, -self.args.token_len:, :]
                batch_y = batch_y[:, -self.args.token_len:, :]
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                n_batches += 1
                
        return total_loss / n_batches
    
    def train_cv(self, folds):
        """Train with K-fold cross-validation."""
        print("Starting K-fold cross-validation training...")
        
        criterion = nn.MSELoss()
        fold_results = []
        
        for fold_idx, (train_ids, val_ids) in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Training Fold {fold_idx + 1}/{len(folds)}")
            print(f"{'='*60}")
            
            # Update args with current fold
            self.args.pop_train_list = train_ids
            self.args.pop_val_list = val_ids
            
            # Reinitialize model for this fold
            self.model = GluLLM.Model(self.args).to(self.device)
            
            # Prepare data
            train_loader, self.scaler = self.prepare_data('train')
            val_loader, _ = self.prepare_data('val')
            
            # Setup optimizer
            optimizer = Adam(self.model.parameters(), 
                            lr=self.args.learning_rate,
                            weight_decay=self.args.weight_decay)
            
            scheduler = CosineAnnealingLR(optimizer, 
                                        T_max=self.args.tmax,
                                        eta_min=1e-8) if self.args.use_scheduler else None
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_path = self.checkpoint_dir / f'checkpoint_fold{fold_idx+1}_{self.args.mn}.pth'
            
            for epoch in range(self.args.train_epochs):
                train_loss = self.train_epoch(train_loader, optimizer, criterion, epoch + 1)
                val_loss = self.evaluate(val_loader, criterion)
                
                if scheduler is not None:
                    scheduler.step()
                
                lr = optimizer.param_groups[0]['lr']
                print(f"  Fold {fold_idx+1} Epoch {epoch+1}/{self.args.train_epochs} - "
                    f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {lr:.8f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"  → Best model for fold {fold_idx+1} saved (val_loss: {val_loss:.6f})")
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model for this fold
            self.model.load_state_dict(torch.load(best_model_path))
            
            fold_results.append({
                'fold': fold_idx + 1,
                'best_val_loss': best_val_loss,
                'final_epoch': epoch + 1
            })
            
            print(f"Fold {fold_idx+1} completed - Best Val Loss: {best_val_loss:.6f}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("Cross-Validation Summary")
        print(f"{'='*60}")
        for result in fold_results:
            print(f"Fold {result['fold']}: Val Loss = {result['best_val_loss']:.6f} "
                f"(stopped at epoch {result['final_epoch']})")
        
        avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
        std_val_loss = np.std([r['best_val_loss'] for r in fold_results])
        print(f"\nAverage Val Loss: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
        
        return fold_results
    
    def predict(self, test_loader):
        """Generate predictions on test set."""
        self.model.eval()
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, prompts in test_loader:
                batch_x = batch_x.to(self.device, dtype=torch.bfloat16)
                batch_y = batch_y.to(self.device, dtype=torch.bfloat16)
                batch_x_mark = batch_x_mark.to(self.device, dtype=torch.bfloat16)
                batch_y_mark = batch_y_mark.to(self.device, dtype=torch.bfloat16)
                    
                pred_len = self.args.test_pred_len
                token_len = self.args.token_len
                n_steps = (pred_len + token_len - 1) // token_len
                
                step_predictions = []
                for step in range(n_steps):
                    if step > 0:
                        pred_x = step_predictions[-1][:, -token_len:, :].detach()
                        batch_x = torch.cat([batch_x[:, token_len:, :], pred_x], dim=1)
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], 
                                                  batch_y_mark[:, step-1:step, :]], dim=1)
                    
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompts)
                    step_predictions.append(outputs[:, -token_len:, :])
                
                pred_y = torch.cat(step_predictions, dim=1)[:, :pred_len, :]
                true_y = batch_y[:, :pred_len, :]
                
                predictions.append(pred_y.cpu().numpy())
                ground_truths.append(true_y.cpu().numpy())
        
        return np.concatenate(predictions), np.concatenate(ground_truths)
    
    def test(self):
        """Run full test pipeline and save results."""
        print("Starting testing...")
        
        # Load model
        model_path = self.checkpoint_dir / f'checkpoint_{self.args.mn}.pth'
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load scaler
        scaler_path = self.checkpoint_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        test_loader, _ = self.prepare_data('test')
        
        # Get dataset lengths
        test_dataset = test_loader.dataset
        if hasattr(test_dataset, 'datasets'):
            dataset_lengths = [len(ds) for ds in test_dataset.datasets]
        else:
            dataset_lengths = [len(test_dataset)]
        
        predictions, ground_truth = self.predict(test_loader)
        
        # Calculate metrics
        mae, mse, rmse, mape, mspe = metric(predictions, ground_truth)
        print(f"Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        if self.scaler is not None:
            pred_metrics(predictions, ground_truth, self.scaler.std)
        
        # Save results
        results = {
            'pred': predictions,
            'true': ground_truth,
            'std': self.scaler.std if self.scaler else 1.0,
            'mean': self.scaler.mean if self.scaler else 0.0,
            'ds_len': dataset_lengths,
        }
        
        results_path = self.results_dir / f'results_{self.args.mn}.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {results_path}")
        return results


def create_kfold_splits(subject_ids, n_folds=5, seed=2024):
    """Create K-fold cross-validation splits."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(subject_ids)):
        train_ids = [subject_ids[i] for i in train_idx]
        val_ids = [subject_ids[i] for i in val_idx]
        folds.append((train_ids, val_ids))
        print(f"Fold {fold_idx + 1}: Train={len(train_ids)}, Val={len(val_ids)}")
    
    return folds

def load_or_create_splits(args):
    """Load existing data splits or create new ones."""
    splits_path = Path(args.results_dir) / f'splits_{args.ds}.pkl'
    
    if splits_path.exists():
        print(f"Loading data splits from {splits_path}")
        with open(splits_path, 'rb') as f:
            train_val_ids, test_ids = pickle.load(f)
    else:
        print("Creating new data splits...")
        
        data_path = Path(args.data_base) / args.ds / 'glullm' / args.mn
        subject_ids = [x[:-17] for x in os.listdir(data_path) 
                       if x.endswith("_embed_prompt.pkl")]
        
        # Split: 80% train+val, 20% test
        np.random.seed(args.seed)
        n_test = int(len(subject_ids) * 0.2)
        test_ids = np.random.choice(subject_ids, n_test, replace=False).tolist()
        train_val_ids = np.setdiff1d(subject_ids, test_ids).tolist()
        
        with open(splits_path, 'wb') as f:
            pickle.dump((train_val_ids, test_ids), f)
        
        print(f"Created splits: Train+Val={len(train_val_ids)}, Test={len(test_ids)}")
    
    return train_val_ids, test_ids


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main execution function."""
    args = args_bglp()
    set_seed(args.seed)
    
    if args.population:
        # Load train+val and test splits
        train_val_ids, test_ids = load_or_create_splits(args)
        
        # Create K-fold splits from train+val
        folds = create_kfold_splits(train_val_ids, n_folds=5, seed=args.seed)
        
        # Set test set
        args.pop_test_list = test_ids
        print(f"Test set: {len(test_ids)} subjects")
    
    # Initialize experiment
    experiment = GlucosePredictionExperiment(args)
    
    # Run K-fold cross-validation
    if args.mode in ['train', 'train_test']:
        fold_results = experiment.train_cv(folds)
        
        # Save CV results
        cv_results_path = experiment.checkpoint_dir / 'cv_results.pkl'
        with open(cv_results_path, 'wb') as f:
            pickle.dump(fold_results, f)
    
    # Test on held-out test set using best fold model
    if args.mode in ['test', 'train_test']:
        # Load best fold based on validation loss
        best_fold = min(fold_results, key=lambda x: x['best_val_loss'])
        best_model_path = experiment.checkpoint_dir / f'checkpoint_fold{best_fold["fold"]}_{args.mn}.pth'
        print(f"\nTesting with best fold {best_fold['fold']} model")
        experiment.model.load_state_dict(torch.load(best_model_path))
        experiment.test()
    
    print("\nExperiment completed successfully!")

if __name__ == '__main__':
    main()