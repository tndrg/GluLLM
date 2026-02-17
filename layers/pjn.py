"""
Projection networks for encoding and decoding time series data.
"""

import torch
import torch.nn as nn
from typing import Literal


class ProjectionNetwork(nn.Module):
    """
    Feedforward network for projecting time series tokens to/from LLM embedding space.
    
    Args:
        input_size: Input feature dimension
        output_size: Output feature dimension  
        num_layers: Number of hidden layers (minimum 1)
        hidden_size: Hidden layer dimension (default: 256)
        dropout_rate: Dropout probability (default: 0.1)
        activation_fn: Activation function type (default: 'tanh')
        use_layer_norm: Whether to apply layer normalization (default: False)
    """
    
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'tanh': nn.Tanh,
        'silu': nn.SiLU,
    }
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 2,
        hidden_size: int = 256,
        dropout_rate: float = 0.1,
        activation_fn: Literal['relu', 'gelu', 'tanh', 'silu'] = 'tanh',
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        
        if activation_fn not in self.ACTIVATIONS:
            raise ValueError(f"activation_fn must be one of {list(self.ACTIVATIONS.keys())}")
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        
        # Build network
        self.network = self._build_network(
            input_size, output_size, num_layers, hidden_size,
            dropout_rate, activation_fn, use_layer_norm
        )
    
    def _build_network(
        self, 
        in_dim: int, 
        out_dim: int, 
        depth: int,
        hidden: int,
        dropout: float,
        activation: str,
        layer_norm: bool,
    ) -> nn.Sequential:
        """Construct the feedforward network."""
        
        act_fn = self.ACTIVATIONS[activation]
        modules = []
        
        # Input projection
        modules.append(nn.Linear(in_dim, hidden))
        if layer_norm:
            modules.append(nn.LayerNorm(hidden))
        modules.append(act_fn())
        modules.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(depth - 1):
            modules.append(nn.Linear(hidden, hidden))
            if layer_norm:
                modules.append(nn.LayerNorm(hidden))
            modules.append(act_fn())
            modules.append(nn.Dropout(dropout))
        
        # Output projection
        modules.append(nn.Linear(hidden, out_dim))
        
        return nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch, seq_len, output_size]
        """
        return self.network(x)
    
    def reset_parameters(self):
        """Reinitialize network parameters."""
        for module in self.network:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()


class ResidualProjection(nn.Module):
    """
    Projection network with residual connections (when input_size == output_size).
    
    Args:
        input_size: Input feature dimension
        output_size: Output feature dimension
        num_layers: Number of residual blocks
        hidden_size: Hidden dimension
        dropout_rate: Dropout probability
        activation_fn: Activation function
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 2,
        hidden_size: int = 256,
        dropout_rate: float = 0.1,
        activation_fn: str = 'gelu',
    ):
        super().__init__()
        
        self.use_residual = (input_size == output_size)
        
        # Input projection if dimensions don't match
        if not self.use_residual:
            self.input_proj = nn.Linear(input_size, output_size)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            self._make_block(output_size, hidden_size, dropout_rate, activation_fn)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(output_size)
    
    def _make_block(self, dim: int, hidden: int, dropout: float, activation: str):
        """Create a single residual block."""
        act_fn = ProjectionNetwork.ACTIVATIONS.get(activation, nn.GELU)
        
        return nn.Sequential(
            nn.Linear(dim, hidden),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        
        # Initial projection if needed
        if not self.use_residual:
            x = self.input_proj(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = x + block(x)
        
        return self.norm(x)