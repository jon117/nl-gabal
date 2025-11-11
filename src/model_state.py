"""
Nested Learning Model with Persistent LSTM States (Tier 3)

This extends the surprise-based model to maintain LSTM hidden states across batches,
enabling true long-range sequential learning.

Key features:
- Persistent LSTM hidden states (h, c)
- Automatic state initialization
- Detached states (prevent BPTT across batches)
- Reset mechanism for document boundaries
- Backward compatible with model_surprise.py
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from .model_surprise import NestedModelWithSurprise


class NestedModelWithState(NestedModelWithSurprise):
    """
    Nested learning model with persistent LSTM states.
    
    Extends NestedModelWithSurprise to maintain LSTM hidden states across batches,
    enabling the model to learn long-range dependencies beyond batch boundaries.
    
    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden layers (default: 2x input_size)
        output_size: Dimension of output (default: input_size)
        track_surprise: Whether to track activations for surprise computation
        reset_on_document: Whether to auto-reset states on new documents
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: Optional[int] = None,
        track_surprise: bool = True,
        reset_on_document: bool = False
    ):
        super().__init__(input_size, hidden_size if hidden_size else input_size * 4)
        
        self.reset_on_document = reset_on_document
        self.track_surprise = track_surprise
        
        # Persistent LSTM hidden states (initialized on first forward pass)
        self.lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        # Track when states were last reset (for debugging)
        self.steps_since_reset = 0
        self.total_resets = 0
    
    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden states.
        
        Args:
            batch_size: Size of the batch
            device: Device to create tensors on (default: same as model)
        
        Returns:
            Tuple of (h_0, c_0) hidden states
        """
        if device is None:
            device = next(self.parameters()).device
        
        # LSTM hidden state: (num_layers, batch_size, hidden_size)
        num_layers = 1  # Our LSTM has 1 layer
        hidden_size = self.input_size  # LSTM hidden size = input_size
        
        h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        
        return (h_0, c_0)
    
    def reset_states(self, batch_size: Optional[int] = None):
        """
        Reset LSTM hidden states.
        
        Call this at document boundaries or when starting a new sequence.
        
        Args:
            batch_size: If provided, initialize new states for this batch size.
                       If None, just clear the states.
        """
        if batch_size is not None:
            device = next(self.parameters()).device
            self.lstm_hidden = self.init_hidden(batch_size, device)
        else:
            self.lstm_hidden = None
        
        self.steps_since_reset = 0
        self.total_resets += 1
    
    def detach_states(self):
        """
        Detach hidden states from computation graph.
        
        This prevents backpropagation through time (BPTT) across batches,
        which would be too expensive and memory-intensive.
        
        Should be called after each batch update.
        """
        if self.lstm_hidden is not None:
            self.lstm_hidden = tuple(h.detach() for h in self.lstm_hidden)
    
    def get_state_info(self) -> Dict[str, any]:
        """
        Get information about current LSTM states.
        
        Returns:
            Dictionary with state information (for debugging/monitoring)
        """
        info = {
            'has_states': self.lstm_hidden is not None,
            'steps_since_reset': self.steps_since_reset,
            'total_resets': self.total_resets
        }
        
        if self.lstm_hidden is not None:
            h, c = self.lstm_hidden
            info['hidden_shape'] = h.shape
            info['hidden_norm'] = h.norm().item()
            info['cell_norm'] = c.norm().item()
            info['batch_size'] = h.size(1)
        
        return info
    
    def forward(
        self,
        x: torch.Tensor,
        compute_surprise: bool = False,
        reset_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, any]]]:
        """
        Forward pass with persistent LSTM states.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            compute_surprise: Whether to track activations for surprise computation
            reset_states: If True, reset LSTM states before this forward pass
        
        Returns:
            logits: Output predictions (batch_size, seq_len, output_size)
            surprise_info: Dictionary of tracked activations (if compute_surprise=True)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Reset states if requested
        if reset_states:
            self.reset_states(batch_size)
        
        # Initialize states if needed (first forward pass or after reset)
        if self.lstm_hidden is None:
            self.lstm_hidden = self.init_hidden(batch_size, device)
        
        # Handle batch size changes (e.g., last batch in epoch)
        current_batch_size = self.lstm_hidden[0].size(1)
        if current_batch_size != batch_size:
            # Adjust hidden states for new batch size
            if batch_size < current_batch_size:
                # Truncate to smaller batch
                self.lstm_hidden = tuple(h[:, :batch_size, :] for h in self.lstm_hidden)
            else:
                # Pad to larger batch (rare, but handle it)
                pad_size = batch_size - current_batch_size
                h, c = self.lstm_hidden
                h_pad = torch.zeros(h.size(0), pad_size, h.size(2), device=device)
                c_pad = torch.zeros(c.size(0), pad_size, c.size(2), device=device)
                self.lstm_hidden = (
                    torch.cat([h, h_pad], dim=1),
                    torch.cat([c, c_pad], dim=1)
                )
        
        # Initialize surprise tracking if requested
        surprise_info = None
        if self.track_surprise and compute_surprise:
            surprise_info = {
                "activations": {},
                "inputs": {}
            }
        
        # Level 1: Fast (LSTM) - with persistent states
        if self.track_surprise and compute_surprise:
            surprise_info['inputs']['level1_fast'] = x
        
        # Forward through LSTM with persistent hidden state
        fast_out, new_hidden = self.level1_fast(x, self.lstm_hidden)
        
        # Update persistent state
        self.lstm_hidden = new_hidden
        
        if self.track_surprise and compute_surprise:
            # Track activation with gradients enabled
            fast_out_tracked = fast_out if fast_out.requires_grad else fast_out.requires_grad_(True)
            surprise_info['activations']['level1_fast'] = fast_out_tracked
        else:
            fast_out_tracked = fast_out
        
        # Level 2: Medium (FFN)
        if self.track_surprise and compute_surprise:
            surprise_info['inputs']['level2_medium'] = fast_out_tracked
        
        medium_out = self.level2_medium(fast_out_tracked)
        
        if self.track_surprise and compute_surprise:
            medium_out_tracked = medium_out if medium_out.requires_grad else medium_out.requires_grad_(True)
            surprise_info['activations']['level2_medium'] = medium_out_tracked
        else:
            medium_out_tracked = medium_out
        
        # Level 3: Slow (FFN)
        if self.track_surprise and compute_surprise:
            surprise_info['inputs']['level3_slow'] = medium_out_tracked
        
        slow_out = self.level3_slow(medium_out_tracked)
        
        if self.track_surprise and compute_surprise:
            surprise_info['activations']['level3_slow'] = slow_out
        
        # Final output layer
        logits = self.output_layer(slow_out)
        
        # Increment step counter
        self.steps_since_reset += 1
        
        return logits, surprise_info
    
    def train_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        compute_surprise: bool = False,
        reset_states: bool = False,
        detach_after: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, any]]]:
        """
        Convenience method for a single training step with state management.
        
        Args:
            x: Input tensor
            targets: Target tensor (for computing loss, not used in forward)
            compute_surprise: Whether to compute surprise signals
            reset_states: Whether to reset LSTM states before this step
            detach_after: Whether to detach states after forward pass (recommended)
        
        Returns:
            logits: Model predictions
            surprise_info: Surprise information (if compute_surprise=True)
        """
        logits, surprise_info = self.forward(x, compute_surprise, reset_states)
        
        # Detach states to prevent BPTT across batches
        if detach_after:
            self.detach_states()
        
        return logits, surprise_info
    
    def get_memory_stats(self) -> Dict[str, any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory information
        """
        stats = {
            'has_persistent_states': self.lstm_hidden is not None,
            'steps_since_reset': self.steps_since_reset,
            'total_resets': self.total_resets
        }
        
        if self.lstm_hidden is not None:
            h, c = self.lstm_hidden
            # Estimate memory usage (float32 = 4 bytes)
            bytes_per_state = h.numel() * 4
            total_bytes = bytes_per_state * 2  # h and c
            stats['state_memory_mb'] = total_bytes / (1024 ** 2)
            stats['state_shape'] = h.shape
        
        return stats


class StatefulTrainingWrapper:
    """
    Wrapper for managing LSTM states during training.
    
    Handles:
    - Document boundary detection
    - Automatic state resets
    - State detachment after updates
    - Statistics tracking
    """
    
    def __init__(
        self,
        model: NestedModelWithState,
        reset_every_n_steps: Optional[int] = None,
        reset_on_new_document: bool = True
    ):
        """
        Initialize wrapper.
        
        Args:
            model: Model with persistent states
            reset_every_n_steps: If set, reset states every N steps (None = never auto-reset)
            reset_on_new_document: If True, reset on document boundaries (needs markers)
        """
        self.model = model
        self.reset_every_n_steps = reset_every_n_steps
        self.reset_on_new_document = reset_on_new_document
        self.step_count = 0
        
        # Statistics
        self.total_steps = 0
        self.total_resets = 0
        self.avg_steps_between_resets = []
    
    def should_reset(self, step: int, is_new_document: bool = False) -> bool:
        """
        Determine if states should be reset.
        
        Args:
            step: Current step number
            is_new_document: Whether this is the start of a new document
        
        Returns:
            True if states should be reset
        """
        # Reset on new document
        if is_new_document and self.reset_on_new_document:
            return True
        
        # Reset every N steps
        if self.reset_every_n_steps is not None:
            if step > 0 and step % self.reset_every_n_steps == 0:
                return True
        
        return False
    
    def forward(
        self,
        x: torch.Tensor,
        compute_surprise: bool = False,
        is_new_document: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, any]]]:
        """
        Forward pass with automatic state management.
        
        Args:
            x: Input tensor
            compute_surprise: Whether to compute surprise
            is_new_document: Whether this is a new document
        
        Returns:
            logits: Model predictions
            surprise_info: Surprise information
        """
        # Check if we should reset
        reset = self.should_reset(self.step_count, is_new_document)
        
        if reset:
            self.model.reset_states(x.size(0))
            self.total_resets += 1
            if len(self.avg_steps_between_resets) > 0:
                self.avg_steps_between_resets.append(
                    self.step_count - sum(self.avg_steps_between_resets)
                )
        
        # Forward pass
        logits, surprise_info = self.model.forward(x, compute_surprise, reset_states=False)
        
        # Detach states after forward (prevent BPTT across batches)
        self.model.detach_states()
        
        self.step_count += 1
        self.total_steps += 1
        
        return logits, surprise_info
    
    def get_stats(self) -> Dict[str, any]:
        """Get training statistics."""
        stats = {
            'total_steps': self.total_steps,
            'total_resets': self.total_resets,
            'current_steps_since_reset': self.model.steps_since_reset
        }
        
        if len(self.avg_steps_between_resets) > 0:
            stats['avg_steps_between_resets'] = sum(self.avg_steps_between_resets) / len(self.avg_steps_between_resets)
        
        return stats
