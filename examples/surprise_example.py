#!/usr/bin/env python3
"""
Example demonstrating Tier 2: Surprise-Based Auxiliary Objectives.

This script shows how to use surprise-based objectives to enhance the
CMS training with auxiliary losses that predict local surprise signals.

Key concepts demonstrated:
1. Initialize model with surprise tracking
2. Setup surprise loss computer with appropriate weights
3. Train with surprise objectives enabled
4. Compare with/without surprise for ablation study
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_surprise import NestedModelWithSurprise
from src.scheduler import ChunkedUpdateScheduler
from src.surprise_loss import SurpriseLossComputer
from src.train_surprise import train_loop_with_surprise
from src.utils import setup_optimizers, set_seed, get_device, create_dummy_data


def main():
    """Run a training example with surprise-based objectives."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device(prefer_cuda=True)
    
    # 1. Initialize Model with Surprise Support
    print("\n" + "="*70)
    print("Initializing NestedModelWithSurprise")
    print("="*70)
    
    model = NestedModelWithSurprise(input_size=768, hidden_size=3072)
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
    
    # 4. Initialize Surprise Loss Computer
    print("\n" + "="*70)
    print("Initializing Surprise Loss Computer")
    print("="*70)
    
    surprise_computer = SurpriseLossComputer(
        loss_weights={
            "level1_fast": 0.3,     # Higher weight for fast level
            "level2_medium": 0.1,   # Lower weight for medium level
            # Note: level3_slow (output layer) doesn't get surprise loss
        },
        gradient_clip_value=10.0,         # Clip extreme gradients
        compute_surprise_every_n_steps=1  # Compute every step (can set to 2-4 to save compute)
    )
    
    # 5. Setup Loss Function
    criterion = nn.MSELoss()
    
    # 6. Create a simple data generator
    def data_generator():
        """Generate dummy batches."""
        return create_dummy_data(
            batch_size=32,
            seq_length=128,
            input_size=768,
            device=device
        )
    
    # 7. Run Training Loop WITH Surprise Objectives
    print("\n" + "="*70)
    print("Training WITH Surprise-Based Objectives")
    print("="*70)
    
    train_loop_with_surprise(
        model=model,
        dataloader=data_generator,
        criterion=criterion,
        optimizers=optimizers,
        scheduler=scheduler,
        surprise_computer=surprise_computer,
        num_steps=1000,
        device=device,
        use_surprise=True,  # ENABLE surprise objectives
        log_interval=50,
        checkpoint_interval=500,
        checkpoint_dir="checkpoints_surprise"
    )
    
    print("\n" + "="*70)
    print("Example Complete!")
    print("="*70)
    print("\nKey Observations:")
    print("- level1_fast updated 1000 times (every step)")
    print("- level2_medium updated ~62 times (every 16 steps)")
    print("- level3_slow updated ~3 times (every 256 steps)")
    print("\nWith surprise objectives:")
    print("- Each layer learns to predict its surprise signal (∇_{y_ℓ} L)")
    print("- Auxiliary losses guide intermediate representations")
    print("- Second-order gradients enable this prediction")
    print("\nThis demonstrates the complete Tier 2 implementation!")
    
    # 8. Optional: Comparison with baseline (no surprise)
    print("\n" + "="*70)
    print("Optional: Run Baseline (No Surprise) for Comparison?")
    print("="*70)
    response = input("Run baseline? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nReinitializing for baseline run...")
        set_seed(42)  # Same seed for fair comparison
        
        model_baseline = NestedModelWithSurprise(input_size=768, hidden_size=3072)
        scheduler_baseline = ChunkedUpdateScheduler(chunk_sizes)
        optimizers_baseline = setup_optimizers(
            model=model_baseline,
            chunk_sizes=chunk_sizes,
            base_lr=base_lr,
            optimizer_type="adam"
        )
        
        print("\n" + "="*70)
        print("Training WITHOUT Surprise (Baseline)")
        print("="*70)
        
        train_loop_with_surprise(
            model=model_baseline,
            dataloader=data_generator,
            criterion=criterion,
            optimizers=optimizers_baseline,
            scheduler=scheduler_baseline,
            surprise_computer=surprise_computer,
            num_steps=1000,
            device=device,
            use_surprise=False,  # DISABLE surprise objectives
            log_interval=50,
            checkpoint_interval=500,
            checkpoint_dir="checkpoints_baseline"
        )
        
        print("\n" + "="*70)
        print("Comparison Complete!")
        print("="*70)
        print("\nYou can now compare:")
        print("- Training dynamics (with vs without surprise)")
        print("- Final loss values")
        print("- Checkpoints for further analysis")


if __name__ == "__main__":
    main()
