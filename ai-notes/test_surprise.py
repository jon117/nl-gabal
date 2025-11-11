#!/usr/bin/env python3
"""
Test script for Tier 2: Surprise-Based Auxiliary Objectives.

This script verifies that:
1. The model can track activations for surprise computation
2. Surprise signals (second-order gradients) can be computed
3. Auxiliary losses are properly calculated
4. The full training loop works with surprise objectives
5. We can toggle surprise on/off for comparison
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_surprise import NestedModelWithSurprise
from src.scheduler import ChunkedUpdateScheduler
from src.surprise_loss import SurpriseLossComputer
from src.train_surprise import train_loop_with_surprise
from src.utils import setup_optimizers, set_seed, get_device, create_dummy_data


def test_forward_with_surprise():
    """Test that forward pass with surprise tracking works."""
    print("\n" + "="*70)
    print("TEST 1: Forward Pass with Surprise Tracking")
    print("="*70)
    
    model = NestedModelWithSurprise(input_size=256, hidden_size=512)
    x = torch.randn(4, 16, 256)
    
    # Test without surprise
    print("Testing forward without surprise tracking...")
    output, surprise_info = model(x, compute_surprise=False)
    assert output.shape == (4, 16, 256), "Output shape incorrect"
    assert surprise_info is None, "Should not return surprise_info when compute_surprise=False"
    print("✓ Forward pass without surprise works")
    
    # Test with surprise
    print("Testing forward with surprise tracking...")
    output, surprise_info = model(x, compute_surprise=True)
    assert output.shape == (4, 16, 256), "Output shape incorrect"
    assert surprise_info is not None, "Should return surprise_info when compute_surprise=True"
    assert "activations" in surprise_info, "Missing activations in surprise_info"
    assert "inputs" in surprise_info, "Missing inputs in surprise_info"
    
    # Check that we have activations for all levels
    expected_levels = ["level1_fast", "level2_medium", "level3_slow"]
    for level in expected_levels:
        assert level in surprise_info["activations"], f"Missing activation for {level}"
        assert level in surprise_info["inputs"], f"Missing input for {level}"
        print(f"  ✓ {level} activation tracked: {surprise_info['activations'][level].shape}")
    
    print("✓ Forward pass with surprise works\n")
    return model, x, output, surprise_info


def test_surprise_signal_computation(model, x, output, surprise_info):
    """Test that surprise signals can be computed."""
    print("="*70)
    print("TEST 2: Surprise Signal Computation")
    print("="*70)
    
    # Compute a dummy loss
    criterion = nn.MSELoss()
    target = torch.randn_like(output)
    loss = criterion(output, target)
    
    print(f"Main loss: {loss.item():.4f}")
    
    # Create surprise computer
    surprise_computer = SurpriseLossComputer(
        loss_weights={"level1_fast": 0.3, "level2_medium": 0.1},
        gradient_clip_value=10.0
    )
    
    # Compute surprise signals
    print("Computing surprise signals...")
    surprise_signals = surprise_computer.compute_surprise_signals(
        loss, 
        surprise_info["activations"]
    )
    
    print(f"Number of surprise signals computed: {len(surprise_signals)}")
    for level_name, signal in surprise_signals.items():
        print(f"  {level_name}: signal shape = {signal.shape}, "
              f"mean = {signal.mean().item():.4f}, "
              f"std = {signal.std().item():.4f}")
        
        # Verify signal has gradient function (second-order)
        assert signal.requires_grad, f"Surprise signal for {level_name} should require grad"
    
    print("✓ Surprise signals computed successfully\n")
    return surprise_computer, surprise_signals


def test_auxiliary_loss_computation(surprise_computer, surprise_info, surprise_signals):
    """Test that auxiliary losses can be computed."""
    print("="*70)
    print("TEST 3: Auxiliary Loss Computation")
    print("="*70)
    
    # Compute auxiliary losses
    print("Computing auxiliary losses...")
    aux_losses = surprise_computer.compute_auxiliary_losses(
        surprise_info,
        surprise_signals
    )
    
    print(f"Number of auxiliary losses: {len(aux_losses)}")
    total_aux_loss = sum(aux_losses.values())
    print(f"Total auxiliary loss: {total_aux_loss.item():.4f}")
    
    for level_name, loss in aux_losses.items():
        print(f"  {level_name}: {loss.item():.4f}")
        assert loss.requires_grad, f"Auxiliary loss for {level_name} should require grad"
    
    print("✓ Auxiliary losses computed successfully\n")
    return total_aux_loss


def test_backward_pass(total_loss):
    """Test that backward pass works with second-order gradients."""
    print("="*70)
    print("TEST 4: Backward Pass with Second-Order Gradients")
    print("="*70)
    
    print("Running backward pass...")
    try:
        total_loss.backward()
        print("✓ Backward pass completed successfully")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        raise
    
    print()


def test_full_training_loop():
    """Test the full training loop with surprise objectives."""
    print("="*70)
    print("TEST 5: Full Training Loop with Surprise")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    device = get_device(prefer_cuda=True)
    
    # Initialize model
    model = NestedModelWithSurprise(input_size=256, hidden_size=512)
    
    # Initialize scheduler
    chunk_sizes = {
        "level1_fast": 1,
        "level2_medium": 8,
        "level3_slow": 16,
    }
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    # Setup optimizers
    base_lr = 1e-4
    optimizers = setup_optimizers(
        model=model,
        chunk_sizes=chunk_sizes,
        base_lr=base_lr,
        optimizer_type="adam"
    )
    
    # Create surprise computer
    surprise_computer = SurpriseLossComputer(
        loss_weights={"level1_fast": 0.3, "level2_medium": 0.1},
        compute_surprise_every_n_steps=2  # Compute every 2 steps to save compute
    )
    
    # Setup loss and data generator
    criterion = nn.MSELoss()
    
    def data_generator():
        return create_dummy_data(
            batch_size=8,
            seq_length=16,
            input_size=256,
            device=device
        )
    
    # Run training
    print("\nRunning 30 steps of training WITH surprise objectives...")
    train_loop_with_surprise(
        model=model,
        dataloader=data_generator,
        criterion=criterion,
        optimizers=optimizers,
        scheduler=scheduler,
        surprise_computer=surprise_computer,
        num_steps=30,
        device=device,
        use_surprise=True,
        log_interval=10
    )
    
    print("✓ Training with surprise completed successfully\n")


def test_comparison_with_without_surprise():
    """Test training with and without surprise for comparison."""
    print("="*70)
    print("TEST 6: Comparison - With vs Without Surprise")
    print("="*70)
    
    set_seed(42)
    device = get_device(prefer_cuda=True)
    
    # Setup (same for both)
    chunk_sizes = {"level1_fast": 1, "level2_medium": 8, "level3_slow": 16}
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    base_lr = 1e-4
    criterion = nn.MSELoss()
    
    def data_generator():
        return create_dummy_data(batch_size=8, seq_length=16, input_size=256, device=device)
    
    # Test WITHOUT surprise
    print("\n[A] Training WITHOUT surprise objectives...")
    model_no_surprise = NestedModelWithSurprise(input_size=256, hidden_size=512)
    optimizers = setup_optimizers(model_no_surprise, chunk_sizes, base_lr)
    surprise_computer = SurpriseLossComputer()
    
    train_loop_with_surprise(
        model=model_no_surprise,
        dataloader=data_generator,
        criterion=criterion,
        optimizers=optimizers,
        scheduler=scheduler,
        surprise_computer=surprise_computer,
        num_steps=20,
        device=device,
        use_surprise=False,  # No surprise
        log_interval=10
    )
    
    # Test WITH surprise
    print("\n[B] Training WITH surprise objectives...")
    set_seed(42)  # Reset seed for fair comparison
    model_with_surprise = NestedModelWithSurprise(input_size=256, hidden_size=512)
    optimizers = setup_optimizers(model_with_surprise, chunk_sizes, base_lr)
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    surprise_computer = SurpriseLossComputer(
        loss_weights={"level1_fast": 0.3, "level2_medium": 0.1}
    )
    
    train_loop_with_surprise(
        model=model_with_surprise,
        dataloader=data_generator,
        criterion=criterion,
        optimizers=optimizers,
        scheduler=scheduler,
        surprise_computer=surprise_computer,
        num_steps=20,
        device=device,
        use_surprise=True,  # With surprise
        log_interval=10
    )
    
    print("✓ Comparison test completed\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TIER 2 TEST SUITE: Surprise-Based Auxiliary Objectives")
    print("="*70)
    print("\nThis test suite verifies the implementation of surprise-based")
    print("objectives as described in the Nested Learning paper.")
    print("="*70)
    
    try:
        # Test 1: Forward pass
        model, x, output, surprise_info = test_forward_with_surprise()
        
        # Test 2: Surprise signal computation
        surprise_computer, surprise_signals = test_surprise_signal_computation(
            model, x, output, surprise_info
        )
        
        # Test 3: Auxiliary loss computation
        total_aux_loss = test_auxiliary_loss_computation(
            surprise_computer, surprise_info, surprise_signals
        )
        
        # Test 4: Backward pass
        criterion = nn.MSELoss()
        target = torch.randn_like(output)
        main_loss = criterion(output, target)
        total_loss = main_loss + total_aux_loss
        test_backward_pass(total_loss)
        
        # Test 5: Full training loop
        test_full_training_loop()
        
        # Test 6: Comparison
        test_comparison_with_without_surprise()
        
        # Summary
        print("="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nTier 2 Implementation Status:")
        print("  ✓ Forward pass with surprise tracking")
        print("  ✓ Surprise signal computation (second-order gradients)")
        print("  ✓ Auxiliary loss computation")
        print("  ✓ Backward pass through second-order gradients")
        print("  ✓ Full training loop with surprise objectives")
        print("  ✓ Toggle surprise on/off")
        print("\nThe surprise-based objectives are fully implemented and working!")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*70)
        raise


if __name__ == "__main__":
    main()
