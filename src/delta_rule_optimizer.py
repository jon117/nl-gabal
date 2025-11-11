"""
Delta-Rule Optimizer (Equation 29 from the paper)

The delta-rule implements a more biologically plausible weight update:
    W_{t+1} = W_t (I - x_t x_t^T) - η ∇_{y_t} L ⊗ x_t

This is different from standard gradient descent:
    W_{t+1} = W_t - η ∇_W L

The key difference is the (I - x_t x_t^T) term, which implements a form of
"synaptic scaling" or "anti-Hebbian" learning that prevents weights from
growing unbounded and creates interference between memories.

This is closely related to:
- Oja's rule for normalized Hebbian learning
- BCM theory of synaptic plasticity
- Fast weights in neural networks
"""

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from typing import Dict, Any, Optional


class DeltaRuleOptimizer(Optimizer):
    """
    Delta-rule optimizer implementing Equation 29 from the paper.
    
    Update rule:
        W_{t+1} = W_t (I - η_decay * x_t x_t^T) - η * ∇_y L ⊗ x_t
    
    Where:
        - W_t: Current weights (hidden_size x input_size)
        - x_t: Input activations
        - ∇_y L: Gradient w.r.t. output
        - η: Learning rate
        - η_decay: Decay rate for the anti-Hebbian term
    
    Note: This requires access to both the input and the gradient w.r.t. output,
    which standard PyTorch optimizers don't have. We need to pass these explicitly.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        decay_rate: float = 1e-4,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        """
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate for gradient term
            decay_rate: Coefficient for anti-Hebbian term (I - x x^T)
            momentum: Momentum coefficient (0 = no momentum)
            weight_decay: L2 regularization coefficient
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if decay_rate < 0.0:
            raise ValueError(f"Invalid decay rate: {decay_rate}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            decay_rate=decay_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
        
        print(f"\n{'='*70}")
        print("DeltaRuleOptimizer initialized")
        print(f"{'='*70}")
        print(f"Learning rate: {lr}")
        print(f"Decay rate (anti-Hebbian): {decay_rate}")
        print(f"Momentum: {momentum}")
        print(f"Weight decay: {weight_decay}")
        print(f"{'='*70}\n")
    
    def step(
        self,
        closure: Optional[callable] = None,
        inputs: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None
    ):
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure to reevaluate the model
            inputs: Input activations x_t (required for delta-rule)
            outputs: Output activations y_t (optional, for debugging)
        
        Returns:
            loss (if closure is provided)
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # If no inputs provided, fall back to standard gradient descent
        if inputs is None:
            return self._standard_step()
        
        # Delta-rule update
        return self._delta_rule_step(inputs, outputs)
    
    def _standard_step(self):
        """
        Standard gradient descent (fallback when inputs not provided).
        
        W_{t+1} = W_t - η * ∇_W L
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    grad = buf
                
                # Update
                p.data.add_(grad, alpha=-lr)
        
        return None
    
    def _delta_rule_step(self, inputs: torch.Tensor, outputs: Optional[torch.Tensor]):
        """
        Delta-rule update with anti-Hebbian term.
        
        For linear layer: W_{t+1} = W_t (I - α x x^T) - η (∇_y L) x^T
        
        Args:
            inputs: Input activations (batch, seq, input_size) or (batch, input_size)
            outputs: Optional output activations (for future use)
        """
        # Process inputs
        if inputs.dim() == 3:
            # (batch, seq, input_size) -> (batch*seq, input_size)
            batch_size, seq_len, input_size = inputs.shape
            inputs = inputs.reshape(-1, input_size)
        elif inputs.dim() == 2:
            # (batch, input_size)
            pass
        else:
            raise ValueError(f"Inputs must be 2D or 3D, got {inputs.dim()}D")
        
        # Average over batch dimension for the anti-Hebbian term
        # x_mean: (input_size,)
        x_mean = inputs.mean(dim=0)
        
        for group in self.param_groups:
            decay_rate = group['decay_rate']
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Only apply delta-rule to 2D weight matrices
                if p.dim() != 2:
                    # For biases and other parameters, use standard update
                    p.data.add_(p.grad.data, alpha=-lr)
                    continue
                
                # W: (output_size, input_size)
                # grad: (output_size, input_size)
                grad = p.grad.data
                
                # Weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Anti-Hebbian term: W (I - α x x^T)
                # Efficient computation: W - α W (x x^T)
                #   = W - α (W x) x^T
                if decay_rate > 0:
                    # Check dimensions: W is (output_size, input_size), x is (input_size,)
                    if p.data.size(1) == x_mean.size(0):  # Correct orientation
                        # Wx_mean: (output_size,)
                        Wx_mean = torch.mv(p.data, x_mean)
                        
                        # Outer product (Wx) ⊗ x: (output_size, input_size)
                        anti_hebbian = torch.outer(Wx_mean, x_mean)
                        
                        # Apply anti-Hebbian decay
                        p.data.sub_(anti_hebbian, alpha=decay_rate)
                    elif p.data.size(0) == x_mean.size(0):  # Transposed
                        # W^T x_mean: (input_size,)
                        Wx_mean = torch.mv(p.data.t(), x_mean)
                        
                        # Outer product x ⊗ (W^T x): (output_size, input_size)
                        anti_hebbian = torch.outer(x_mean, Wx_mean)
                        
                        # Apply anti-Hebbian decay
                        p.data.sub_(anti_hebbian, alpha=decay_rate)
                    # else: dimensions don't match, skip anti-Hebbian term
                
                # Standard gradient term with momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(grad)
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    grad = buf
                
                # Apply gradient update
                p.data.add_(grad, alpha=-lr)
        
        return None


class AdamDeltaRule(Optimizer):
    """
    Hybrid optimizer combining Adam's adaptive learning rates with delta-rule.
    
    This combines:
    1. Adam's first and second moment estimates for adaptive LR
    2. Delta-rule's anti-Hebbian term for biological plausibility
    
    Update:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        g_hat = m_t / (sqrt(v_t) + ε)
        W_{t+1} = W_t (I - α x x^T) - η * g_hat
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        decay_rate: float = 1e-4,
        weight_decay: float = 0.0
    ):
        """
        Args:
            params: Parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added for numerical stability
            decay_rate: Anti-Hebbian decay rate
            weight_decay: L2 penalty
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if decay_rate < 0.0:
            raise ValueError(f"Invalid decay rate: {decay_rate}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            decay_rate=decay_rate,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
        
        print(f"\n{'='*70}")
        print("AdamDeltaRule initialized")
        print(f"{'='*70}")
        print(f"Learning rate: {lr}")
        print(f"Betas: {betas}")
        print(f"Epsilon: {eps}")
        print(f"Decay rate (anti-Hebbian): {decay_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"{'='*70}\n")
    
    def step(
        self,
        closure: Optional[callable] = None,
        inputs: Optional[torch.Tensor] = None
    ):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        # Process inputs if provided
        if inputs is not None:
            if inputs.dim() == 3:
                inputs = inputs.reshape(-1, inputs.size(-1))
            x_mean = inputs.mean(dim=0)
        else:
            x_mean = None
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = (bias_correction2 ** 0.5)
                
                # Compute adaptive gradient
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                adaptive_grad = exp_avg / denom
                
                # Anti-Hebbian term (only for 2D weight matrices)
                if p.dim() == 2 and x_mean is not None and group['decay_rate'] > 0:
                    # Check dimensions
                    if p.data.size(1) == x_mean.size(0):  # Correct orientation
                        Wx_mean = torch.mv(p.data, x_mean)
                        anti_hebbian = torch.outer(Wx_mean, x_mean)
                        p.data.sub_(anti_hebbian, alpha=group['decay_rate'])
                    elif p.data.size(0) == x_mean.size(0):  # Transposed
                        Wx_mean = torch.mv(p.data.t(), x_mean)
                        anti_hebbian = torch.outer(x_mean, Wx_mean)
                        p.data.sub_(anti_hebbian, alpha=group['decay_rate'])
                
                # Apply update
                p.data.add_(adaptive_grad, alpha=-step_size)
        
        return loss


def setup_delta_rule_optimizers(
    model,
    chunk_sizes: Dict[str, int],
    base_lr: float = 1e-3,
    optimizer_type: str = "delta_adam",  # "delta_sgd" or "delta_adam"
    decay_rate: float = 1e-4,
    **kwargs
) -> Dict[str, Optimizer]:
    """
    Setup delta-rule optimizers for each level with scaled learning rates.
    
    Args:
        model: Model with .levels attribute
        chunk_sizes: Dict of level_name -> chunk_size
        base_lr: Base learning rate
        optimizer_type: "delta_sgd" or "delta_adam"
        decay_rate: Anti-Hebbian decay rate
        **kwargs: Additional optimizer arguments
    
    Returns:
        Dict of level_name -> optimizer
    """
    optimizers = {}
    
    print("\n" + "="*70)
    print("Setting up Delta-Rule optimizers with scaled learning rates")
    print("="*70)
    print(f"{'Level':<20} {'Chunk Size':<12} {'Base LR':<12} {'Scaled LR':<12}")
    print("-"*70)
    
    for level_name, module in model.levels.items():
        chunk_size = chunk_sizes[level_name]
        scaled_lr = base_lr / chunk_size
        
        if optimizer_type == "delta_sgd":
            optimizer = DeltaRuleOptimizer(
                module.parameters(),
                lr=scaled_lr,
                decay_rate=decay_rate,
                **kwargs
            )
        elif optimizer_type == "delta_adam":
            optimizer = AdamDeltaRule(
                module.parameters(),
                lr=scaled_lr,
                decay_rate=decay_rate,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        optimizers[level_name] = optimizer
        
        print(f"{level_name:<20} {chunk_size:<12} {base_lr:<12.2e} {scaled_lr:<12.2e}")
    
    print("="*70 + "\n")
    
    return optimizers


# Example usage
if __name__ == "__main__":
    print("Testing Delta-Rule Optimizer...")
    
    # Create a simple linear layer
    linear = torch.nn.Linear(10, 5)
    
    # Test DeltaRuleOptimizer
    print("\n1. Testing DeltaRuleOptimizer:")
    opt = DeltaRuleOptimizer(linear.parameters(), lr=0.01, decay_rate=0.001)
    
    # Forward pass
    x = torch.randn(32, 10)  # batch of 32
    y = linear(x)
    loss = y.sum()
    
    # Backward
    loss.backward()
    
    # Step with inputs
    opt.step(inputs=x)
    
    print("✓ DeltaRuleOptimizer step complete")
    
    # Test AdamDeltaRule
    print("\n2. Testing AdamDeltaRule:")
    linear2 = torch.nn.Linear(10, 5)
    opt2 = AdamDeltaRule(linear2.parameters(), lr=0.01, decay_rate=0.001)
    
    y2 = linear2(x)
    loss2 = y2.sum()
    loss2.backward()
    opt2.step(inputs=x)
    
    print("✓ AdamDeltaRule step complete")
    
    print("\n✅ All tests passed!")
