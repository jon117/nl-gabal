"""
Training loop implementation for Nested Learning (CMS).

This module implements the core training logic with:
- Step-aligned gradient accumulation
- Selective gradient zeroing per level
- Memory-efficient updates
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from tqdm import tqdm

from .model import NestedModel
from .scheduler import ChunkedUpdateScheduler


def train_step(
    model: NestedModel,
    data: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    scheduler: ChunkedUpdateScheduler,
    global_step: int,
    device: torch.device
) -> float:
    """
    Perform a single training step with selective gradient accumulation.
    
    This is the core of the Nested Learning implementation. Key steps:
    1. Forward & backward pass (calculates gradients for ALL parameters)
    2. Selective update & gradient zeroing based on scheduler logic
    
    Args:
        model: The NestedModel instance
        data: Input data tensor
        targets: Target labels
        criterion: Loss function
        optimizers: Dictionary of optimizers (one per level)
        scheduler: ChunkedUpdateScheduler instance
        global_step: Current training step (1-indexed)
        device: Device to run on
    
    Returns:
        Loss value for this step
    """
    model.train()
    
    # Move data to device
    data = data.to(device)
    targets = targets.to(device)
    
    # 1. FORWARD & BACKWARD PASS
    # This single pass calculates gradients for ALL parameters.
    # Gradients for levels that don't update will accumulate automatically.
    output = model(data)
    
    # For sequence outputs, we need to reshape for the loss calculation
    # output shape: (batch, seq_len, input_size)
    # We'll add a simple projection head for demonstration
    # In practice, you'd have a proper task-specific head
    batch_size, seq_len, _ = output.shape
    
    # Simple mean pooling and linear projection for classification
    # (This is just for demonstration - replace with your actual task)
    output_pooled = output.mean(dim=1)  # (batch, input_size)
    
    # For now, let's just use a simple MSE loss on the output
    # In a real scenario, you'd have a proper classification/regression head
    loss = criterion(output, data)  # Reconstruction-style loss for demo
    
    loss.backward()
    
    # 2. SELECTIVE UPDATE & GRADIENT ZEROING
    # Iterate through each level and check if it should update
    for level_name, module in model.levels.items():
        
        # Ask the scheduler if it's time to update this level
        if scheduler.should_update(level_name, global_step):
            
            # The .grad attributes now contain the sum of gradients over the chunk.
            # The learning rate has been scaled by 1/chunk_size to compensate.
            # Perform the optimizer step.
            optimizers[level_name].step()
            
            # Notify scheduler that this level was updated
            scheduler.mark_updated(level_name, global_step)
            
            # CRITICAL: Zero the gradients ONLY for the level that was just updated.
            # This is the key to selective accumulation - other levels keep their gradients.
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.zero_()
    
    # The selective zeroing ensures that gradients for level2_medium and level3_slow
    # persist across steps until their update condition is met.
    
    return loss.item()


def train_loop(
    model: NestedModel,
    dataloader: Callable,  # Function that yields (data, targets)
    criterion: nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    scheduler: ChunkedUpdateScheduler,
    num_steps: int,
    device: torch.device,
    log_interval: int = 100,
    checkpoint_interval: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    start_step: int = 0
):
    """
    Main training loop with multi-timescale updates.
    
    Args:
        model: NestedModel instance
        dataloader: Function that yields (data, targets) batches
        criterion: Loss function
        optimizers: Dictionary of optimizers
        scheduler: ChunkedUpdateScheduler
        num_steps: Total number of training steps
        device: Device to train on
        log_interval: How often to log progress
        checkpoint_interval: How often to save checkpoints (None to disable)
        checkpoint_dir: Directory to save checkpoints
        start_step: Starting step (for resuming training)
    """
    from .utils import save_checkpoint
    
    model.to(device)
    global_step = start_step
    
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}")
    print(f"Total steps: {num_steps}")
    print(f"Starting from step: {start_step}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Training loop
    progress_bar = tqdm(range(start_step, num_steps), desc="Training")
    running_loss = 0.0
    
    for step in progress_bar:
        global_step = step + 1  # 1-indexed
        
        # Get next batch
        data, targets = dataloader()
        
        # Perform training step
        loss = train_step(
            model=model,
            data=data,
            targets=targets,
            criterion=criterion,
            optimizers=optimizers,
            scheduler=scheduler,
            global_step=global_step,
            device=device
        )
        
        running_loss += loss
        
        # Logging
        if global_step % log_interval == 0:
            avg_loss = running_loss / log_interval
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Print scheduler stats
            if global_step % (log_interval * 10) == 0:
                scheduler.print_stats(global_step)
            
            running_loss = 0.0
        
        # Checkpointing
        if checkpoint_interval and global_step % checkpoint_interval == 0:
            if checkpoint_dir:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_step_{global_step}.pt"
                save_checkpoint(
                    model=model,
                    optimizers=optimizers,
                    scheduler=scheduler,
                    global_step=global_step,
                    save_path=checkpoint_path
                )
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    scheduler.print_stats(global_step)


def evaluate(
    model: NestedModel,
    dataloader: Callable,
    criterion: nn.Module,
    device: torch.device,
    num_steps: int = 100
) -> float:
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        dataloader: Function that yields (data, targets) batches
        criterion: Loss function
        device: Device to evaluate on
        num_steps: Number of evaluation steps
    
    Returns:
        Average loss over evaluation steps
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_steps):
            data, targets = dataloader()
            data = data.to(device)
            targets = targets.to(device)
            
            output = model(data)
            loss = criterion(output, data)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_steps
    return avg_loss
