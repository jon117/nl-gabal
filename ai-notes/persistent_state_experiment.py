#!/usr/bin/env python3
"""
Experiment: Test Persistent LSTM States on Long Sequences

This experiment compares models with and without persistent LSTM states
on longer sequences to demonstrate the benefit of cross-batch memory.

Key comparisons:
1. Baseline (no states) - resets every batch
2. Persistent states - maintains memory across batches
3. With surprise objectives
4. Different sequence lengths

Expected: Persistent states should significantly improve performance on
longer sequences where cross-batch dependencies matter.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_state import NestedModelWithState
from src.model_surprise import NestedModelWithSurprise
from src.scheduler import ChunkedUpdateScheduler
from src.surprise_loss import SurpriseLossComputer

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

# Download WikiText-2
def download_wikitext2():
    """Download WikiText-2 dataset."""
    import urllib.request
    
    url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    cache_file = Path(__file__).parent / "wikitext2_train.txt"
    
    if cache_file.exists():
        print(f"Using cached WikiText-2: {cache_file}")
    else:
        print(f"Downloading WikiText-2 from {url}...")
        urllib.request.urlretrieve(url, cache_file)
        print(f"Saved to {cache_file}")
    
    with open(cache_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text

# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

# Create batches
def create_batches(data, batch_size, seq_len):
    """Create batches of sequences."""
    n_batches = len(data) // (batch_size * seq_len)
    data = data[:n_batches * batch_size * seq_len]
    
    # Reshape into batches
    data = torch.tensor(data).view(batch_size, -1)
    
    batches = []
    for i in range(0, data.size(1) - seq_len, seq_len):
        x = data[:, i:i+seq_len]
        y = data[:, i+1:i+seq_len+1]
        batches.append((x, y))
    
    return batches

def setup_optimizers(model, scheduler, base_lr=3e-4):
    """Setup optimizers with scaled learning rates."""
    optimizers = {}
    
    print("\n" + "="*70)
    print("Setting up optimizers with scaled learning rates")
    print("="*70)
    print(f"{'Level':<20} {'Chunk Size':<12} {'Base LR':<12} {'Scaled LR':<12}")
    print("-"*70)
    
    for level_name, module in model.levels.items():
        chunk_size = scheduler.chunk_sizes[level_name]
        scaled_lr = base_lr / chunk_size
        
        optimizers[level_name] = torch.optim.Adam(
            module.parameters(),
            lr=scaled_lr
        )
        
        print(f"{level_name:<20} {chunk_size:<12} {base_lr:<12.2e} {scaled_lr:<12.2e}")
    
    print("="*70)
    return optimizers

def train_model(
    model,
    batches,
    scheduler,
    optimizers,
    criterion,
    device,
    num_steps=2000,
    use_surprise=False,
    surprise_computer=None,
    persistent_states=False,
    detach_every=1,
    log_every=200
):
    """Train model."""
    model.train()
    losses = []
    start_time = time.time()
    
    step = 0
    batch_idx = 0
    
    # Reset states at start if using persistent states
    if persistent_states and hasattr(model, 'reset_states'):
        model.reset_states(batches[0][0].size(0))
    
    while step < num_steps:
        # Get batch
        x, y = batches[batch_idx % len(batches)]
        x, y = x.to(device), y.to(device)
        
        # Embed inputs
        x_emb = model.embedding(x)
        
        # Forward pass
        compute_surprise = use_surprise and surprise_computer is not None
        logits, surprise_info = model(x_emb, compute_surprise=compute_surprise)
        
        # Compute main loss
        main_loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss = main_loss
        aux_loss = 0.0
        
        # Add surprise loss if using it
        if compute_surprise:
            aux_loss, _ = surprise_computer.compute(main_loss, surprise_info)
            total_loss = main_loss + aux_loss
        
        # Backward
        total_loss.backward()
        
        # Update parameters that should be updated at this step
        for level_name, module in model.levels.items():
            if scheduler.should_update(level_name, step):
                optimizers[level_name].step()
                scheduler.mark_updated(level_name, step)
                module.zero_grad()
        
        # Detach states periodically if using persistent states
        if persistent_states and hasattr(model, 'detach_states'):
            if step % detach_every == 0:
                model.detach_states()
        
        # Track loss
        losses.append(main_loss.item())
        
        # Log progress
        if (step + 1) % log_every == 0:
            avg_loss = sum(losses[-log_every:]) / log_every
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            elapsed = time.time() - start_time
            
            state_info = ""
            if persistent_states and hasattr(model, 'get_state_info'):
                info = model.get_state_info()
                state_info = f" | States: {info['steps_since_reset']} steps"
            
            print(f"  Step {step+1}/{num_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                  f"Aux: {aux_loss if isinstance(aux_loss, float) else aux_loss.item():.4f}{state_info} | Time: {elapsed:.1f}s")
        
        step += 1
        batch_idx += 1
    
    # Final perplexity
    final_loss = sum(losses[-100:]) / min(100, len(losses))
    final_ppl = torch.exp(torch.tensor(final_loss)).item()
    
    total_time = time.time() - start_time
    
    return final_ppl, losses, total_time

def run_experiment(
    name,
    batches,
    vocab_size,
    device,
    use_surprise=False,
    persistent_states=False,
    seq_len=256,
    num_steps=2000
):
    """Run a single experiment configuration."""
    print("\n" + "="*70)
    print(f"EXPERIMENT: {name}")
    print("="*70)
    print(f"Persistent states: {persistent_states}")
    print(f"Use surprise: {use_surprise}")
    print(f"Sequence length: {seq_len}")
    print(f"Steps: {num_steps}")
    print("="*70)
    
    set_seed(42)
    
    # Model
    if persistent_states:
        model = NestedModelWithState(
            input_size=256,
            hidden_size=512,
            track_surprise=use_surprise
        )
    else:
        model = NestedModelWithSurprise(
            input_size=256,
            hidden_size=512
        )
    
    # Move model to device first
    model = model.to(device)
    
    # Add embedding and output layers (after moving to device)
    model.embedding = nn.Embedding(vocab_size, 256).to(device)
    model.output_layer = nn.Linear(256, vocab_size).to(device)
    
    # Scheduler
    chunk_sizes = {
        "level1_fast": 1,
        "level2_medium": 16,
        "level3_slow": 256
    }
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    # Optimizers
    optimizers = setup_optimizers(model, scheduler, base_lr=3e-4)
    
    # Surprise computer
    surprise_computer = None
    if use_surprise:
        surprise_computer = SurpriseLossComputer(
            loss_weights={"level1_fast": 0.05, "level2_medium": 0.01}
        )
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print(f"\nTraining for {num_steps} steps...")
    final_ppl, losses, total_time = train_model(
        model=model,
        batches=batches,
        scheduler=scheduler,
        optimizers=optimizers,
        criterion=criterion,
        device=device,
        num_steps=num_steps,
        use_surprise=use_surprise,
        surprise_computer=surprise_computer,
        persistent_states=persistent_states,
        detach_every=1,
        log_every=200
    )
    
    print("\n" + "="*70)
    print(f"RESULTS: {name}")
    print("="*70)
    print(f"Final perplexity: {final_ppl:.2f}")
    print(f"Total time: {total_time:.1f}s")
    print("="*70)
    
    return {
        'name': name,
        'final_ppl': final_ppl,
        'total_time': total_time,
        'persistent_states': persistent_states,
        'use_surprise': use_surprise,
        'seq_len': seq_len,
        'losses': losses
    }

def main():
    print("="*70)
    print("PERSISTENT LSTM STATES EXPERIMENT")
    print("="*70)
    print("\nTesting impact of persistent LSTM states on long sequences")
    print("="*70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Download data
    print("\nDownloading WikiText-2...")
    text = download_wikitext2()
    print(f"Downloaded {len(text):,} characters")
    
    # Tokenize
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary: {tokenizer.vocab_size} characters")
    
    tokens = tokenizer.encode(text)
    print(f"Dataset: {len(tokens):,} tokens")
    
    # Experiment configurations
    configs = [
        # Baseline: No persistent states, no surprise
        {
            'name': '1_baseline_reset_every_batch',
            'persistent_states': False,
            'use_surprise': False,
            'seq_len': 256,
            'num_steps': 2000
        },
        # Persistent states only
        {
            'name': '2_persistent_states_only',
            'persistent_states': True,
            'use_surprise': False,
            'seq_len': 256,
            'num_steps': 2000
        },
        # Persistent states + surprise
        {
            'name': '3_persistent_plus_surprise',
            'persistent_states': True,
            'use_surprise': True,
            'seq_len': 256,
            'num_steps': 2000
        },
        # Longer sequences with persistent states
        {
            'name': '4_persistent_long_sequences',
            'persistent_states': True,
            'use_surprise': False,
            'seq_len': 512,
            'num_steps': 2000
        }
    ]
    
    results = []
    
    for config in configs:
        # Create batches for this configuration
        batches = create_batches(
            tokens,
            batch_size=32,
            seq_len=config['seq_len']
        )
        
        print(f"\nCreated {len(batches)} batches of size 32 x {config['seq_len']}")
        
        # Run experiment
        result = run_experiment(
            name=config['name'],
            batches=batches,
            vocab_size=tokenizer.vocab_size,
            device=device,
            use_surprise=config['use_surprise'],
            persistent_states=config['persistent_states'],
            seq_len=config['seq_len'],
            num_steps=config['num_steps']
        )
        
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Configuration':<40} {'Final PPL':<12} {'PPL Δ':<12} {'Time (s)':<10}")
    print("-"*70)
    
    baseline_ppl = results[0]['final_ppl']
    
    for result in results:
        ppl_delta = baseline_ppl - result['final_ppl']
        name_short = result['name'].replace('_', ' ').title()
        print(f"{name_short:<40} {result['final_ppl']:<12.2f} {ppl_delta:>+11.2f} {result['total_time']:<10.1f}")
    
    print("-"*70)
    
    # Find best
    best = min(results, key=lambda r: r['final_ppl'])
    print(f"\n🏆 Best configuration: {best['name']}")
    print(f"   Final PPL: {best['final_ppl']:.2f}")
    print(f"   Improvement: {baseline_ppl - best['final_ppl']:+.2f} PPL")
    
    # Save results
    output_file = Path(__file__).parent / "persistent_state_results.json"
    
    # Remove losses (too large for JSON)
    for r in results:
        if 'losses' in r:
            del r['losses']
    
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'baseline_ppl': baseline_ppl,
            'best_config': best['name'],
            'best_ppl': best['final_ppl'],
            'improvement': baseline_ppl - best['final_ppl']
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
