import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.model_info import LLM_CONFIGS
from layers.pjn import ProjectionNetwork

class Model(nn.Module):
    """
    GluLLM for blood glucose prediction.
    
    Encodes glucose time series into LLM embedding space, processes with frozen LLM,
    then decodes predictions back to glucose values.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self._print_device_info()
        
        self.config = config
        self.device = torch.device('cuda:0')
        self.patch_size = config.token_len
        self.model_name = config.mn
        self.cache_dir = config.cache_dir
        
        # Load LLM and get configuration
        self.llm, self.tokenizer, self.embed_dim = self._load_language_model()
        
        # Projection networks
        self.ts_encoder = self._build_encoder()
        self.ts_decoder = self._build_decoder()
        
        # Optional: temporal feature mixing
        self.use_tdi_mix = config.mix_embeds
        if self.use_tdi_mix:
            self.mix_scale = nn.Parameter(
                torch.ones([], dtype=torch.bfloat16, device=self.device)
            )
        
        # Optional: text prompt conditioning
        self.use_prompts = config.use_prompt
        
    def _print_device_info(self):
        """Print available GPU devices."""
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    def _load_language_model(self):
        """Load pretrained language model with frozen weights."""
        
        if self.model_name not in LLM_CONFIGS:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Available: {list(LLM_CONFIGS.keys())}"
            )
        
        model_cfg = LLM_CONFIGS[self.model_name]
        checkpoint = model_cfg['checkpoint']
        embed_dim = model_cfg['embed_dim']
        
        print(f"Loading {self.model_name} from {checkpoint}")
        
        # Load model
        llm = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        
        # Freeze all LLM parameters
        for param in llm.parameters():
            param.requires_grad = False
        
        # Load tokenizer (use Llama tokenizer for OpenELM)
        tokenizer_name = checkpoint if self.model_name != 'openelm' else 'meta-llama/Llama-2-7b-hf'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        return llm, tokenizer, embed_dim
    
    def _build_encoder(self):
        """Build time series encoder (glucose → LLM space)."""
        
        if self.config.mlp_hidden_layers == 0:
            print("Using linear encoder")
            encoder = nn.Linear(self.patch_size, self.embed_dim)
        else:
            print(f"Using {self.config.mlp_hidden_layers}-layer MLP encoder")
            encoder = ProjectionNetwork(
                input_size=self.patch_size,
                output_size=self.embed_dim,
                num_layers=self.config.mlp_hidden_layers,
                hidden_size=self.config.mlp_hidden_dim,
                dropout_rate=self.config.dropout,
                activation_fn=self.config.mlp_activation,
            )
        
        return encoder.to(self.device).bfloat16()
    
    def _build_decoder(self):
        """Build prediction decoder (LLM space → glucose)."""
        
        # Place decoder on last GPU if multi-GPU
        decoder_device = f"cuda:{torch.cuda.device_count() - 1}"
        
        if self.config.mlp_hidden_layers == 0:
            print("Using linear decoder")
            decoder = nn.Linear(self.embed_dim, self.patch_size)
        else:
            print(f"Using {self.config.mlp_hidden_layers}-layer MLP decoder")
            decoder = ProjectionNetwork(
                input_size=self.embed_dim,
                output_size=self.patch_size,
                num_layers=self.config.mlp_hidden_layers,
                hidden_size=self.config.mlp_hidden_dim,
                dropout_rate=self.config.dropout,
                activation_fn=self.config.mlp_activation,
            )
        
        return decoder.to(decoder_device).bfloat16()
    
    def _normalize_input(self, x):
        """Instance normalization: zero mean, unit variance."""
        mean = x.mean(dim=1, keepdim=True).detach()
        x_centered = x - mean
        std = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_normalized = x_centered / std
        return x_normalized, mean, std
    
    def _patchify(self, x):
        """Convert time series to non-overlapping patches."""
        batch_size, seq_len, n_vars = x.shape
        
        # Reshape: [batch, seq, vars] → [batch*vars, seq]
        x = x.permute(0, 2, 1).reshape(-1, seq_len)
        
        # Extract patches: [batch*vars, n_patches, patch_size]
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        
        return patches, batch_size, n_vars
    
    def _tdi(self, patches, tdi_embeds=None):
        """
        Integrate time dependent Interpreter (TDI) with glucose time series embeddings.
        
        TDI captures contextual events like insulin delivery that influence real-time glucose dynamics.
        These embeddings are pre-computed using LLMs and stored locally to avoid repeated 
        inference overhead during training.
        
        Args:
            patches: Glucose time series patches [batch*vars, n_patches, patch_size]
            tdi_embeds: Pre-computed TDI embeddings [batch*vars, n_patches, embed_dim] (optional)
            
        Returns:
            Combined embeddings [batch*vars, n_patches, embed_dim]
        """
        # patches: [batch*vars, n_patches, patch_size] → [batch*vars, n_patches, embed_dim]
        embeddings = self.ts_encoder(patches)

        if self.use_tdi_mix and tdi_embeds is not None:
            # L2 normalize
            embeddings = embeddings / embeddings.norm(dim=2, keepdim=True)
            tdi_embeds = tdi_embeds / tdi_embeds.norm(dim=2, keepdim=True)
            # Weighted combination
            embeddings = embeddings + self.mix_scale * tdi_embeds
        
        return embeddings
    
    def _add_prompt_conditioning(self, embeddings, prompts):
        """Prepend demographics prompt embeddings to time series embeddings."""
        if not self.use_prompts or prompts is None:
            return embeddings
        
        # Tokenize prompts
        tokens = self.tokenizer(prompts, return_tensors="pt").input_ids
        # Get embeddings
        prompt_embeds = self.llm.get_input_embeddings()(tokens.to(self.device))
        # Concatenate: [prompts, time_series]
        return torch.cat([prompt_embeds, embeddings], dim=1)
    
    def _process_with_llm(self, embeddings):
        """Process embeddings through frozen LLM."""
        # Different LLMs have different module names
        if self.model_name in ['openelm', 'gpt']:
            outputs = self.llm.transformer(inputs_embeds=embeddings)[0]
        else:
            outputs = self.llm.model(inputs_embeds=embeddings)[0]
        
        return outputs
    
    def _decode_prediction(self, llm_outputs, batch_size, n_vars):
        """Decode LLM outputs back to glucose predictions."""
        # Take last token's representation
        last_hidden = llm_outputs[:, -1:, :]
        
        # Move to decoder device and decode
        decoder_device = self.ts_decoder.network[0].weight.device
        decoded = self.ts_decoder(last_hidden.to(decoder_device))
        
        # Reshape: [batch*vars, 1, patch_size] → [batch, patch_size, vars]
        decoded = decoded.reshape(batch_size, n_vars, -1).permute(0, 2, 1)
        
        return decoded
    
    def _denormalize_output(self, x, mean, std):
        """Reverse instance normalization."""
        n_patches = x.shape[1]
        
        # Broadcast mean and std
        std = std.to(x.device)
        mean = mean.to(x.device)
        
        std_expanded = std[:, 0, :].unsqueeze(1).repeat(1, n_patches, 1)
        mean_expanded = mean[:, 0, :].unsqueeze(1).repeat(1, n_patches, 1)
        
        return x * std_expanded + mean_expanded
    
    def forecast(self, x_input, tdi_embeds=None, prompts=None):
        """
        Generate glucose forecast.
        
        Args:
            x_input: Input glucose values [batch, seq_len, n_vars]
            tdi_embeds: TDI embeddings [batch*vars, n_patches, embed_dim]
            prompts: demographics prompts (list of strings)
            
        Returns:
            Predicted glucose values [batch, pred_len, n_vars]
        """
        # Step 1: Normalize
        x_norm, mean, std = self._normalize_input(x_input)
        
        # Step 2: Patchify
        patches, batch_size, n_vars = self._patchify(x_norm)
        
        # Step 3: Encode to LLM space
        embeddings = self._tdi(patches, tdi_embeds)
        
        # Step 4: Add prompt conditioning
        llm_input = self._add_prompt_conditioning(embeddings, prompts)
        
        # Step 5: Process with LLM
        llm_output = self._process_with_llm(llm_input)
        
        # Step 6: Decode prediction
        prediction = self._decode_prediction(llm_output, batch_size, n_vars)
        
        # Step 7: Denormalize
        prediction = self._denormalize_output(prediction, mean, std)
        
        return prediction
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, prompts):
        return self.forecast(x_enc, x_mark_enc, prompts)