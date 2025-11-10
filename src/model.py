"""
NestedModel: A multi-level architecture for hierarchical learning.

This model implements the Continuum Memory System (CMS) from the Nested Learning paper.
Parameters are organized into distinct "levels" that will be updated at different frequencies
by the ChunkedUpdateScheduler.

Design Philosophy:
- The model is a standard PyTorch nn.Module, unaware of the nested update logic
- Layers are grouped into a self.levels dictionary for programmatic access
- The forward pass is a simple, uninterrupted computational graph (no .detach() calls)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class NestedModel(nn.Module):
    """
    A feed-forward network with parameters logically grouped into learning levels.
    
    Architecture:
        - Level 1 (Fast): LSTM for rapid, short-term pattern adaptation
        - Level 2 (Medium): Feed-forward network for mid-term feature extraction
        - Level 3 (Slow): Feed-forward network for long-term structural learning
    
    Args:
        input_size: Dimensionality of input features (default: 768)
        hidden_size: Size of hidden layers in feed-forward networks (default: 3072)
    """
    
    def __init__(self, input_size: int = 768, hidden_size: int = 3072):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Level 1: Fast - LSTM for rapid adaptation to recent patterns
        # Updates every step (chunk_size=1)
        self.level1_fast = nn.LSTM(
            input_size=input_size,
            hidden_size=input_size,
            batch_first=True
        )
        
        # Level 2: Medium - Feed-forward network for mid-term patterns
        # Updates every 16 steps (chunk_size=16)
        self.level2_medium = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Level 3: Slow - Feed-forward network for long-term structure
        # Updates every 256 steps (chunk_size=256)
        self.level3_slow = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # KEY: Store levels in a dictionary for easy access by scheduler and optimizer setup
        # This enables programmatic iteration over levels without manual tracking
        self.levels: Dict[str, nn.Module] = {
            "level1_fast": self.level1_fast,
            "level2_medium": self.level2_medium,
            "level3_slow": self.level3_slow,
        }
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using standard PyTorch initialization."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                # LSTM weights are already initialized by PyTorch
                pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all levels sequentially.
        
        Note: We're omitting residual connections for clarity in this Tier 1 implementation.
        The computational graph is uninterrupted - no .detach() calls are needed.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, input_size)
        """
        # Level 1: Fast LSTM processing
        # lstm output shape: (batch_size, seq_len, input_size)
        fast_out, _ = self.level1_fast(x)
        
        # Level 2: Medium-term feed-forward processing
        medium_out = self.level2_medium(fast_out)
        
        # Level 3: Slow, long-term feed-forward processing
        slow_out = self.level3_slow(medium_out)
        
        return slow_out
    
    def get_level_names(self):
        """Returns the names of all learning levels."""
        return list(self.levels.keys())
    
    def get_level_params(self, level_name: str):
        """
        Get parameters for a specific level.
        
        Args:
            level_name: Name of the level (e.g., 'level1_fast')
        
        Returns:
            Iterator over parameters of the specified level
        """
        if level_name not in self.levels:
            raise ValueError(f"Unknown level: {level_name}. Available levels: {self.get_level_names()}")
        return self.levels[level_name].parameters()
    
    def count_parameters(self, level_name: Optional[str] = None) -> int:
        """
        Count trainable parameters.
        
        Args:
            level_name: If provided, count parameters only for this level.
                       If None, count all parameters.
        
        Returns:
            Number of trainable parameters
        """
        if level_name is None:
            # Count all parameters
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            # Count parameters for specific level
            return sum(p.numel() for p in self.get_level_params(level_name) if p.requires_grad)
    
    def print_model_info(self):
        """Print detailed information about the model architecture."""
        print("=" * 70)
        print("NestedModel Architecture")
        print("=" * 70)
        print(f"Input size: {self.input_size}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"\nParameter counts by level:")
        total_params = 0
        for level_name in self.get_level_names():
            level_params = self.count_parameters(level_name)
            total_params += level_params
            print(f"  {level_name:20s}: {level_params:,} parameters")
        print(f"  {'TOTAL':20s}: {total_params:,} parameters")
        print("=" * 70)
