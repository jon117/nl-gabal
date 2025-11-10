#!/usr/bin/env python3
"""
Simple example demonstrating the Nested Learning (CMS) implementation.

This script shows how to:
1. Initialize the model and scheduler
2. Setup optimizers with scaled learning rates
3. Run the training loop with selective gradient accumulation
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import NestedModel
from src.scheduler import ChunkedUpdateScheduler
from src.train import train_loop
from src.utils import setup_optimizers, set_seed, get_device, create_dummy_data


def main():
    """Run a simple training example."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device(prefer_cuda=True)
    
    # 1. Initialize Model
    print("\n" + "="*70)
    print("Initializing NestedModel")
    print("="*70)
    
    model = NestedModel(input_size=768, hidden_size=3072)
    model.print_model_info()
    
    # 2. Initialize Scheduler
    print("\n" + "="*70)
    print("Initializing ChunkedUpdateScheduler")
    print("="*70)
    
    chunk_sizes = {
        "level1_fast": 1,      # Updates every step
        "level2_medium": 16,   # Updates every 16 steps
        "level3_slow": 256,    # Updates every 256 steps
    }
    
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    # 3. Setup Optimizers with Scaled Learning Rates
    base_lr = 1e-4
    optimizers = setup_optimizers(
        model=model,
        chunk_sizes=chunk_sizes,
        base_lr=base_lr,
        optimizer_type="adam",
        weight_decay=0.0
    )
    
    # 4. Setup Loss Function
    criterion = nn.MSELoss()
    
    # 5. Create a simple data generator (dummy data for demonstration)
    def data_generator():
        """Generate dummy batches."""
        return create_dummy_data(
            batch_size=32,
            seq_length=128,
            input_size=768,
            device=device
        )
    
    # 6. Run Training Loop
    print("\n" + "="*70)
    print("Starting Training Loop")
    print("="*70)
    
    train_loop(
        model=model,
        dataloader=data_generator,
        criterion=criterion,
        optimizers=optimizers,
        scheduler=scheduler,
        num_steps=1000,
        device=device,
        log_interval=50,
        checkpoint_interval=500,
        checkpoint_dir="checkpoints"
    )
    
    print("\n" + "="*70)
    print("Example Complete!")
    print("="*70)
    print("\nKey Observations:")
    print("- level1_fast updated 1000 times (every step)")
    print("- level2_medium updated ~62 times (every 16 steps)")
    print("- level3_slow updated ~3 times (every 256 steps)")
    print("\nThis demonstrates the multi-timescale learning hierarchy!")


if __name__ == "__main__":
    main()
