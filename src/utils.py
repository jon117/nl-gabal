"""
Utility functions for Nested Learning implementation.

Includes configuration loading, optimizer setup, and logging utilities.
"""

import yaml
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_optimizers(
    model: nn.Module,
    chunk_sizes: Dict[str, int],
    base_lr: float = 1e-4,
    optimizer_type: str = "adam",
    weight_decay: float = 0.0
) -> Dict[str, torch.optim.Optimizer]:
    """
    Setup optimizers for each learning level with scaled learning rates.
    
    CRITICAL: Learning rates are scaled by 1/chunk_size to account for gradient
    accumulation. This approximates using a smaller LR on a larger effective batch size.
    
    For example:
    - level1_fast (chunk_size=1): LR = base_lr / 1 = 1e-4
    - level2_medium (chunk_size=16): LR = base_lr / 16 = 6.25e-6
    - level3_slow (chunk_size=256): LR = base_lr / 256 = 3.9e-7
    
    Args:
        model: The NestedModel instance with .levels dictionary
        chunk_sizes: Dictionary mapping level names to chunk sizes
        base_lr: Base learning rate (will be scaled per level)
        optimizer_type: Type of optimizer ("adam", "sgd", "adamw")
        weight_decay: Weight decay (L2 regularization)
    
    Returns:
        Dictionary mapping level names to optimizer instances
    """
    optimizers = {}
    
    print(f"\n{'='*70}")
    print("Setting up optimizers with scaled learning rates")
    print(f"{'='*70}")
    print(f"{'Level':<20} {'Chunk Size':<12} {'Base LR':<12} {'Scaled LR':<12}")
    print(f"{'-'*70}")
    
    for level_name, module in model.levels.items():
        chunk_size = chunk_sizes.get(level_name, 1)
        
        # Scale the learning rate by 1/chunk_size
        scaled_lr = base_lr / chunk_size
        
        # Create optimizer based on type
        if optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                module.parameters(),
                lr=scaled_lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                module.parameters(),
                lr=scaled_lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                module.parameters(),
                lr=scaled_lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        optimizers[level_name] = optimizer
        
        print(f"{level_name:<20} {chunk_size:<12} {base_lr:<12.2e} {scaled_lr:<12.2e}")
    
    print(f"{'='*70}\n")
    
    return optimizers


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def count_model_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the device to use for training.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
    
    Returns:
        torch.device instance
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
    return device


def create_dummy_data(
    batch_size: int,
    seq_length: int,
    input_size: int,
    num_classes: int = 10,
    device: Optional[torch.device] = None
):
    """
    Create dummy data for testing/demonstration purposes.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        input_size: Input feature dimension
        num_classes: Number of output classes
        device: Device to create tensors on
    
    Returns:
        Tuple of (data, targets)
    """
    if device is None:
        device = torch.device("cpu")
    
    # Random input data
    data = torch.randn(batch_size, seq_length, input_size, device=device)
    
    # Random classification targets
    targets = torch.randint(0, num_classes, (batch_size, seq_length), device=device)
    
    return data, targets


def save_checkpoint(
    model: nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    scheduler: Any,
    global_step: int,
    save_path: str
):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizers: Dictionary of optimizers
        scheduler: The update scheduler
        global_step: Current training step
        save_path: Path to save the checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dicts': {
            name: opt.state_dict() for name, opt in optimizers.items()
        },
        'scheduler_state': {
            'chunk_sizes': scheduler.chunk_sizes,
            'last_update_step': scheduler.last_update_step,
            'update_counts': scheduler.update_counts,
        },
        'global_step': global_step,
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    scheduler: Any
) -> int:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load state into
        optimizers: Dictionary of optimizers to load state into
        scheduler: Scheduler to load state into
    
    Returns:
        global_step from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    for name, opt in optimizers.items():
        if name in checkpoint['optimizer_state_dicts']:
            opt.load_state_dict(checkpoint['optimizer_state_dicts'][name])
    
    scheduler.last_update_step = checkpoint['scheduler_state']['last_update_step']
    scheduler.update_counts = checkpoint['scheduler_state']['update_counts']
    
    global_step = checkpoint['global_step']
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Resuming from step: {global_step}")
    
    return global_step
