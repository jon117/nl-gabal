"""
Surprise-Based Auxiliary Loss Computation.

This module implements the core insight from the Nested Learning paper:
each layer learns to predict its "surprise signal" (the gradient of the final
loss w.r.t. its activation).

Theoretical Background:
    Equation 27 from the paper:
        min_W ||W x_t - ∇_{y_t} L(W_t; x_t)||²_2
    
    This says: minimize the L2 distance between the layer's output (W x_t)
    and the surprise signal (∇_{y_t} L).
    
    In practice:
        1. Compute main loss L
        2. Compute surprise signals: ∇_{y_ℓ} L for each intermediate level ℓ
        3. Compute auxiliary loss: ||y_ℓ - ∇_{y_ℓ} L||²_2
        4. Total loss = L + Σ(λ_ℓ * L_aux_ℓ)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class SurpriseLossComputer:
    """
    Computes surprise-based auxiliary losses following the paper's formulation.
    
    The surprise signal at level ℓ is: ∇_{y_ℓ} L
    The auxiliary loss is: ||y_ℓ - ∇_{y_ℓ} L||²_2
    
    This creates a second-order gradient computation:
        - First order: standard backprop from final loss
        - Second order: gradient of loss w.r.t. intermediate activations
    
    Args:
        loss_weights: Dict mapping level names to their auxiliary loss weights
        gradient_clip_value: Maximum absolute value for surprise gradients (for stability)
        compute_surprise_every_n_steps: Compute surprise only every N steps (to save compute)
    """
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        gradient_clip_value: float = 10.0,
        compute_surprise_every_n_steps: int = 1
    ):
        """
        Initialize the surprise loss computer.
        
        Args:
            loss_weights: Dict mapping level names to their loss weights.
                         Typical values: 0.1 - 0.3 for intermediate levels.
            gradient_clip_value: Clip surprise gradients to this range for stability.
            compute_surprise_every_n_steps: Only compute surprise every N steps
                                           (1 = every step, 2 = every other step, etc.)
        """
        self.loss_weights = loss_weights or {
            "level1_fast": 0.3,     # Higher weight for fast level (updates frequently)
            "level2_medium": 0.1,   # Lower weight for medium level
            # Note: level3_slow typically doesn't get surprise loss (it's the output)
        }
        
        self.gradient_clip_value = gradient_clip_value
        self.compute_surprise_every_n_steps = compute_surprise_every_n_steps
        self.step_counter = 0
        
        # Use MSE loss for the auxiliary objective
        self.mse = nn.MSELoss()
        
        print(f"\n{'='*70}")
        print("SurpriseLossComputer initialized")
        print(f"{'='*70}")
        print(f"Loss weights: {self.loss_weights}")
        print(f"Gradient clip value: {self.gradient_clip_value}")
        print(f"Compute surprise every {self.compute_surprise_every_n_steps} steps")
        print(f"{'='*70}\n")
    
    def should_compute_surprise(self) -> bool:
        """
        Determine if we should compute surprise this step.
        
        Returns:
            True if we should compute surprise, False otherwise
        """
        self.step_counter += 1
        return (self.step_counter % self.compute_surprise_every_n_steps) == 0
    
    def compute_surprise_signals(
        self,
        main_loss: torch.Tensor,
        activations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the surprise signal (∇_{y_ℓ} L) for each level.
        
        This is the gradient of the main loss w.r.t. each intermediate activation.
        Computing these gradients requires create_graph=True to enable second-order
        gradients (backprop through the gradient computation itself).
        
        Args:
            main_loss: The final task loss (scalar tensor with grad_fn)
            activations: Dict of activations from surprise_info
                        Keys: level names, Values: activation tensors
            
        Returns:
            surprise_signals: Dict mapping level names to their surprise signals
                            Each signal has the same shape as its corresponding activation
        """
        surprise_signals = {}
        
        for level_name, activation in activations.items():
            # Skip the final level - it's the output, no auxiliary loss needed
            if level_name == "level3_slow":
                continue
            
            # Skip if this level doesn't have a weight (not used for surprise)
            if level_name not in self.loss_weights:
                continue
            
            try:
                # CRITICAL: Compute ∇_{y_ℓ} L
                # This is the gradient of the loss w.r.t. this activation
                # 
                # create_graph=True: Enable second-order gradients
                #   (we need to backprop through this gradient computation)
                # 
                # retain_graph=True: Don't free the computation graph
                #   (we'll compute multiple gradients for different levels)
                # 
                # only_inputs=True: Only compute gradients for the specified input
                surprise = torch.autograd.grad(
                    outputs=main_loss,
                    inputs=activation,
                    create_graph=True,     # Enable backprop through gradient
                    retain_graph=True,     # Keep graph for next gradient computation
                    only_inputs=True,      # Only compute for this activation
                    allow_unused=False,    # Raise error if activation not used
                )[0]
                
                # SAFEGUARD: Clip extreme gradients for numerical stability
                # Second-order gradients can sometimes explode
                if self.gradient_clip_value is not None:
                    surprise = torch.clamp(
                        surprise,
                        min=-self.gradient_clip_value,
                        max=self.gradient_clip_value
                    )
                
                surprise_signals[level_name] = surprise
                
            except RuntimeError as e:
                # If gradient computation fails (e.g., activation not in graph),
                # fall back to zero surprise
                print(f"Warning: Failed to compute surprise for {level_name}: {e}")
                surprise_signals[level_name] = torch.zeros_like(activation)
        
        return surprise_signals
    
    def compute_auxiliary_losses(
        self,
        surprise_info: Dict,
        surprise_signals: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the auxiliary loss for each level.
        
        Implements Equation 27: L_aux = ||y_ℓ - ∇_{y_ℓ} L||²_2
        
        The layer's activation y_ℓ should match the surprise signal ∇_{y_ℓ} L.
        This makes the layer learn to predict how it should change.
        
        Args:
            surprise_info: Dict from model forward pass containing 'activations' and 'inputs'
            surprise_signals: Dict of computed surprise signals from compute_surprise_signals
            
        Returns:
            aux_losses: Dict mapping level names to their weighted auxiliary losses
        """
        aux_losses = {}
        activations = surprise_info["activations"]
        
        for level_name, surprise in surprise_signals.items():
            activation = activations[level_name]
            
            # The auxiliary objective: predict the surprise signal
            # This is the heart of Equation 27 from the paper
            # 
            # CRITICAL: .detach() on the surprise signal
            # Why? We want gradients to flow through 'activation' but NOT through
            # 'surprise'. If we don't detach, we get third-order gradients:
            #   First order: main_loss.backward()
            #   Second order: surprise = grad(main_loss, activation)
            #   Third order: would occur from backprop through surprise
            # 
            # Third-order gradients are almost never beneficial and make training
            # extremely unstable.
            aux_loss = self.mse(activation, surprise.detach())
            
            # Weight the loss according to the specified weight for this level
            weighted_loss = self.loss_weights.get(level_name, 0.0) * aux_loss
            aux_losses[level_name] = weighted_loss
        
        return aux_losses
    
    def compute(
        self,
        main_loss: torch.Tensor,
        surprise_info: Dict,
        force_compute: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Complete surprise-based auxiliary loss computation.
        
        This is the main entry point for computing surprise losses.
        
        Args:
            main_loss: Final task loss (scalar tensor)
            surprise_info: Dict from model.forward(..., compute_surprise=True)
            force_compute: If True, compute even if should_compute_surprise() is False
            
        Returns:
            total_aux_loss: Sum of weighted auxiliary losses (scalar tensor)
            aux_losses_dict: Individual losses for logging (dict of scalar tensors)
        """
        # Check if we should compute surprise this step
        if not force_compute and not self.should_compute_surprise():
            # Return zero loss and empty dict
            return torch.tensor(0.0, device=main_loss.device), {}
        
        # Step 1: Compute surprise signals (∇_{y_ℓ} L)
        surprise_signals = self.compute_surprise_signals(
            main_loss, 
            surprise_info["activations"]
        )
        
        # If no surprise signals computed (all skipped), return zero
        if not surprise_signals:
            return torch.tensor(0.0, device=main_loss.device), {}
        
        # Step 2: Compute auxiliary losses (||y_ℓ - ∇_{y_ℓ} L||²_2)
        aux_losses = self.compute_auxiliary_losses(surprise_info, surprise_signals)
        
        # Step 3: Sum for total auxiliary loss
        if aux_losses:
            total_aux_loss = sum(aux_losses.values())
        else:
            total_aux_loss = torch.tensor(0.0, device=main_loss.device)
        
        return total_aux_loss, aux_losses
    
    def reset_counter(self):
        """Reset the step counter (useful when starting a new epoch)."""
        self.step_counter = 0
