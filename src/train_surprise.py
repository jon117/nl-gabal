"""
Training loop with surprise-based auxiliary objectives (Tier 2).

This module extends the basic training loop to support surprise-based losses.
The key difference is that we compute second-order gradients to enable the
surprise signal computation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from tqdm import tqdm

from .model_surprise import NestedModelWithSurprise
from .scheduler import ChunkedUpdateScheduler
from .surprise_loss import SurpriseLossComputer


def train_step_with_surprise(
    model: NestedModelWithSurprise,
    data: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    scheduler: ChunkedUpdateScheduler,
    surprise_computer: SurpriseLossComputer,
    global_step: int,
    device: torch.device,
    use_surprise: bool = True
) -> tuple:
    """
    Perform a single training step with surprise-based auxiliary objectives.
    
    This extends the basic train_step with:
    1. Forward pass with surprise tracking (if use_surprise=True)
    2. Computation of surprise signals (second-order gradients)
    3. Auxiliary losses based on surprise prediction
    4. Combined loss backpropagation
    
    Args:
        model: The NestedModelWithSurprise instance
        data: Input data tensor
        targets: Target labels
        criterion: Loss function
        optimizers: Dictionary of optimizers (one per level)
        scheduler: ChunkedUpdateScheduler instance
        surprise_computer: SurpriseLossComputer instance
        global_step: Current training step (1-indexed)
        device: Device to run on
        use_surprise: If True, compute surprise-based auxiliary losses
    
    Returns:
        Tuple of (main_loss, aux_loss, aux_losses_dict)
    """
    model.train()
    
    # Move data to device
    data = data.to(device)
    targets = targets.to(device)
    
    # 1. FORWARD PASS (with or without surprise tracking)
    if use_surprise:
        output, surprise_info = model(data, compute_surprise=True)
    else:
        output, _ = model(data, compute_surprise=False)
    
    # 2. COMPUTE MAIN LOSS
    # For sequence outputs, we use reconstruction loss
    # In practice, you'd have a proper task-specific head
    loss = criterion(output, data)
    
    # 3. COMPUTE SURPRISE-BASED AUXILIARY LOSSES
    aux_loss = torch.tensor(0.0, device=device)
    aux_losses_dict = {}
    
    if use_surprise and surprise_info is not None:
        # This computes second-order gradients:
        # - First order: gradients w.r.t. parameters
        # - Second order: gradients w.r.t. intermediate activations
        aux_loss, aux_losses_dict = surprise_computer.compute(
            main_loss=loss,
            surprise_info=surprise_info
        )
    
    # 4. TOTAL LOSS
    total_loss = loss + aux_loss
    
    # 5. BACKWARD PASS
    # CRITICAL: This backpropagates through second-order gradients
    # due to create_graph=True in surprise signal computation
    total_loss.backward()
    
    # 6. SELECTIVE UPDATE & GRADIENT ZEROING
    # Same logic as Tier 1, but now with surprise-enhanced gradients
    for level_name, module in model.levels.items():
        
        # Ask the scheduler if it's time to update this level
        if scheduler.should_update(level_name, global_step):
            
            # Perform the optimizer step
            optimizers[level_name].step()
            
            # Notify scheduler that this level was updated
            scheduler.mark_updated(level_name, global_step)
            
            # CRITICAL: Zero the gradients ONLY for the level that was just updated
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.zero_()
    
    return loss.item(), aux_loss.item(), aux_losses_dict


def train_loop_with_surprise(
    model: NestedModelWithSurprise,
    dataloader: Callable,
    criterion: nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    scheduler: ChunkedUpdateScheduler,
    surprise_computer: SurpriseLossComputer,
    num_steps: int,
    device: torch.device,
    use_surprise: bool = True,
    log_interval: int = 100,
    checkpoint_interval: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    start_step: int = 0
):
    """
    Main training loop with surprise-based auxiliary objectives.
    
    Args:
        model: NestedModelWithSurprise instance
        dataloader: Function that yields (data, targets) batches
        criterion: Loss function
        optimizers: Dictionary of optimizers
        scheduler: ChunkedUpdateScheduler
        surprise_computer: SurpriseLossComputer instance
        num_steps: Total number of training steps
        device: Device to train on
        use_surprise: If True, use surprise-based auxiliary losses
        log_interval: How often to log progress
        checkpoint_interval: How often to save checkpoints (None to disable)
        checkpoint_dir: Directory to save checkpoints
        start_step: Starting step (for resuming training)
    """
    from .utils import save_checkpoint
    
    model.to(device)
    global_step = start_step
    
    print(f"\n{'='*70}")
    print("Starting Training with Surprise-Based Objectives")
    print(f"{'='*70}")
    print(f"Total steps: {num_steps}")
    print(f"Starting from step: {start_step}")
    print(f"Device: {device}")
    print(f"Use surprise: {use_surprise}")
    if use_surprise:
        print(f"Surprise loss weights: {surprise_computer.loss_weights}")
    print(f"{'='*70}\n")
    
    # Training loop
    progress_bar = tqdm(range(start_step, num_steps), desc="Training")
    running_loss = 0.0
    running_aux_loss = 0.0
    running_aux_losses = {}
    
    for step in progress_bar:
        global_step = step + 1  # 1-indexed
        
        # Get next batch
        data, targets = dataloader()
        
        # Perform training step
        main_loss, aux_loss, aux_losses_dict = train_step_with_surprise(
            model=model,
            data=data,
            targets=targets,
            criterion=criterion,
            optimizers=optimizers,
            scheduler=scheduler,
            surprise_computer=surprise_computer,
            global_step=global_step,
            device=device,
            use_surprise=use_surprise
        )
        
        running_loss += main_loss
        running_aux_loss += aux_loss
        
        # Accumulate aux losses for each level
        for level_name, loss_val in aux_losses_dict.items():
            if level_name not in running_aux_losses:
                running_aux_losses[level_name] = 0.0
            running_aux_losses[level_name] += loss_val.item()
        
        # Logging
        if global_step % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_aux_loss = running_aux_loss / log_interval
            
            # Build postfix dict for progress bar
            postfix = {'main_loss': f'{avg_loss:.4f}'}
            if use_surprise and avg_aux_loss > 0:
                postfix['aux_loss'] = f'{avg_aux_loss:.4f}'
            
            progress_bar.set_postfix(postfix)
            
            # Print detailed stats every 10 intervals
            if global_step % (log_interval * 10) == 0:
                print(f"\n{'='*70}")
                print(f"Step {global_step}")
                print(f"{'='*70}")
                print(f"Main loss: {avg_loss:.4f}")
                if use_surprise:
                    print(f"Total aux loss: {avg_aux_loss:.4f}")
                    if running_aux_losses:
                        print("Aux losses by level:")
                        for level_name, total_loss in running_aux_losses.items():
                            avg_level_loss = total_loss / log_interval
                            print(f"  {level_name:20s}: {avg_level_loss:.4f}")
                
                # Print scheduler stats
                scheduler.print_stats(global_step)
                print(f"{'='*70}\n")
            
            running_loss = 0.0
            running_aux_loss = 0.0
            running_aux_losses = {}
        
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


def evaluate_with_surprise(
    model: NestedModelWithSurprise,
    dataloader: Callable,
    criterion: nn.Module,
    device: torch.device,
    num_steps: int = 100
) -> tuple:
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        dataloader: Function that yields (data, targets) batches
        criterion: Loss function
        device: Device to evaluate on
        num_steps: Number of evaluation steps
    
    Returns:
        Tuple of (avg_loss, model)
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_steps):
            data, targets = dataloader()
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward without surprise tracking (more efficient for eval)
            output, _ = model(data, compute_surprise=False)
            loss = criterion(output, data)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_steps
    return avg_loss
