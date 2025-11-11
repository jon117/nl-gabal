"""
NestedModel with Surprise-Based Objectives (Tier 2).

This module extends the basic NestedModel to support surprise-based auxiliary losses
as described in the Nested Learning paper.

Theoretical Foundation:
    The paper's insight is that each layer is an associative memory that learns to map
    inputs to their "local surprise signal". The surprise signal is the gradient of the
    final loss w.r.t. that layer's activation: ∇_{y_ℓ} L
    
    The auxiliary objective (Equation 27) is:
        L_aux = ||y_ℓ - ∇_{y_ℓ} L||²_2
    
    This makes each layer learn to predict how it should change based on the final loss.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class NestedModelWithSurprise(nn.Module):
    """
    A multi-level architecture with surprise-based auxiliary objectives.
    
    This model extends the basic NestedModel to track intermediate activations
    so we can compute surprise signals (gradients w.r.t. those activations).
    
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
        
        # Output layer (projects from input_size to input_size by default)
        # Can be replaced for specific tasks (e.g., classification, language modeling)
        self.output_layer = nn.Linear(input_size, input_size)
        
        # KEY: Store levels in a dictionary for easy access by scheduler and optimizer setup
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
    
    def forward(
        self, 
        x: torch.Tensor, 
        compute_surprise: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with optional surprise signal computation.
        
        When compute_surprise=True, we track intermediate activations with
        requires_grad=True so we can compute gradients w.r.t. them later.
        This enables the surprise-based auxiliary objectives.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            compute_surprise: If True, retains activations for surprise computation
        
        Returns:
            output: Final output tensor
            surprise_info: Dict containing activations and inputs for each level
                          (None if compute_surprise=False)
        """
        if not compute_surprise:
            # Standard forward pass (more efficient, no surprise tracking)
            fast_out, _ = self.level1_fast(x)
            medium_out = self.level2_medium(fast_out)
            slow_out = self.level3_slow(medium_out)
            logits = self.output_layer(slow_out)
            return logits, None
        
        # SURPRISE-TRACKING FORWARD PASS
        # We need to store:
        # 1. Inputs to each level (x_ℓ in the paper)
        # 2. Activations from each level (y_ℓ in the paper)
        
        inputs = {}
        activations = {}
        
        # Level 1: Fast LSTM processing
        inputs["level1_fast"] = x
        fast_out, _ = self.level1_fast(x)
        
        # CRITICAL: Register this activation for gradient computation
        # We need gradients w.r.t. this tensor to compute the surprise signal
        if fast_out.requires_grad:
            # If already requires_grad (training mode), just store it
            fast_out_tracked = fast_out
        else:
            # Explicitly enable gradient tracking
            fast_out_tracked = fast_out.requires_grad_(True)
        
        activations["level1_fast"] = fast_out_tracked
        
        # Level 2: Medium-term feed-forward processing
        inputs["level2_medium"] = fast_out_tracked
        medium_out = self.level2_medium(fast_out_tracked)
        
        # Track medium output
        if medium_out.requires_grad:
            medium_out_tracked = medium_out
        else:
            medium_out_tracked = medium_out.requires_grad_(True)
        
        activations["level2_medium"] = medium_out_tracked
        
        # Level 3: Slow, long-term feed-forward processing
        inputs["level3_slow"] = medium_out_tracked
        slow_out = self.level3_slow(medium_out_tracked)
        
        # Always track final output (we compute gradients w.r.t. it for the main loss)
        activations["level3_slow"] = slow_out
        
        # Final output layer
        logits = self.output_layer(slow_out)
        
        # Package up the information needed for surprise computation
        surprise_info = {
            "activations": activations,  # The y_ℓ values
            "inputs": inputs,            # The x_ℓ values
        }
        
        return logits, surprise_info
    
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
        print("NestedModelWithSurprise Architecture")
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
