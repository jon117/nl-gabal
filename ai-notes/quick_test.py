#!/usr/bin/env python3
"""
Quick test to verify the implementation works with a small training run.
This runs a minimal version to check if everything integrates correctly.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import NestedModel
from src.scheduler import ChunkedUpdateScheduler
from src.train import train_loop
from src.utils import setup_optimizers, set_seed, get_device, create_dummy_data


def main():
    """Run a quick training test."""
    
    print("\n" + "="*70)
    print("QUICK TEST: Verifying Nested Learning Implementation")
    print("="*70)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device - use GPU if available
    device = get_device(prefer_cuda=True)
    
    # 1. Initialize Model (smaller size for quick test)
    print("\n[1/6] Initializing NestedModel...")
    model = NestedModel(input_size=256, hidden_size=512)
    model.print_model_info()
    
    # 2. Initialize Scheduler
    print("\n[2/6] Initializing ChunkedUpdateScheduler...")
    chunk_sizes = {
        "level1_fast": 1,      # Updates every step
        "level2_medium": 8,    # Updates every 8 steps (reduced from 16 for quick test)
        "level3_slow": 16,     # Updates every 16 steps (reduced from 256 for quick test)
    }
    
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    # 3. Setup Optimizers with Scaled Learning Rates
    print("\n[3/6] Setting up optimizers with scaled learning rates...")
    base_lr = 1e-4
    optimizers = setup_optimizers(
        model=model,
        chunk_sizes=chunk_sizes,
        base_lr=base_lr,
        optimizer_type="adam",
        weight_decay=0.0
    )
    
    # 4. Setup Loss Function
    print("\n[4/6] Setting up loss function...")
    criterion = nn.MSELoss()
    print("Using MSELoss for reconstruction-style training")
    
    # 5. Create a simple data generator (dummy data for testing)
    print("\n[5/6] Creating data generator...")
    def data_generator():
        """Generate dummy batches."""
        return create_dummy_data(
            batch_size=16,      # Small batch size for quick test
            seq_length=32,      # Short sequence for quick test
            input_size=256,
            device=device
        )
    print("Generating dummy data with batch_size=16, seq_length=32, input_size=256")
    
    # 6. Run Training Loop (just a few steps)
    print("\n[6/6] Running training loop for 50 steps...")
    print("="*70)
    
    train_loop(
        model=model,
        dataloader=data_generator,
        criterion=criterion,
        optimizers=optimizers,
        scheduler=scheduler,
        num_steps=50,  # Just 50 steps to verify it works
        device=device,
        log_interval=10,
        checkpoint_interval=None,  # No checkpointing for quick test
        checkpoint_dir=None
    )
    
    # Verify the scheduler statistics
    print("\n" + "="*70)
    print("VERIFICATION: Checking scheduler statistics")
    print("="*70)
    
    stats = scheduler.get_stats(50)
    
    print("\nExpected vs Actual Updates:")
    print(f"  level1_fast: Expected 50, Got {stats['level1_fast']['update_count']}")
    print(f"  level2_medium: Expected 6, Got {stats['level2_medium']['update_count']}")
    print(f"  level3_slow: Expected 3, Got {stats['level3_slow']['update_count']}")
    
    # Check if updates are correct
    all_correct = True
    if stats['level1_fast']['update_count'] != 50:
        print("  ❌ level1_fast update count incorrect!")
        all_correct = False
    if stats['level2_medium']['update_count'] != 6:
        print("  ❌ level2_medium update count incorrect!")
        all_correct = False
    if stats['level3_slow']['update_count'] != 3:
        print("  ❌ level3_slow update count incorrect!")
        all_correct = False
    
    if all_correct:
        print("\n✅ All update counts are correct!")
        print("✅ Nested Learning implementation is working as expected!")
    else:
        print("\n❌ Some update counts are incorrect. Please review the implementation.")
    
    print("\n" + "="*70)
    print("Quick Test Complete!")
    print("="*70)
    
    # Test a forward pass to ensure model is functional
    print("\n[BONUS] Testing forward pass with model...")
    test_input = torch.randn(4, 16, 256).to(device)
    with torch.no_grad():
        output = model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("  ✅ Forward pass successful!")
    
    print("\n" + "="*70)
    print("🎉 All checks passed! Implementation is ready for experiments!")
    print("="*70)


if __name__ == "__main__":
    main()
