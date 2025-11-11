#!/usr/bin/env python3
"""
Real Data Experiment: Testing on WikiText-2

This script downloads real language modeling data and runs proper experiments
to see if surprise objectives help on structured, real-world tasks.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import time
import json
from collections import defaultdict
import requests
import zipfile
import io

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_surprise import NestedModelWithSurprise
from src.scheduler import ChunkedUpdateScheduler
from src.surprise_loss import SurpriseLossComputer
from src.train_surprise import train_step_with_surprise
from src.utils import setup_optimizers, set_seed, get_device


class SimpleTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, text):
        # Get unique characters
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        print(f"Vocabulary size: {self.vocab_size} characters")
    
    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char.get(i, '?') for i in indices])


class TextDataset:
    """Simple text dataset for language modeling."""
    
    def __init__(self, text, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize entire text
        self.tokens = tokenizer.encode(text)
        print(f"Dataset size: {len(self.tokens):,} tokens")
        
    def get_batch(self, batch_size, device):
        """Get a random batch of sequences."""
        # Random starting positions
        max_start = len(self.tokens) - self.seq_length - 1
        starts = torch.randint(0, max_start, (batch_size,))
        
        # Gather sequences
        x = torch.zeros(batch_size, self.seq_length, dtype=torch.long)
        y = torch.zeros(batch_size, self.seq_length, dtype=torch.long)
        
        for i, start in enumerate(starts):
            x[i] = torch.tensor(self.tokens[start:start+self.seq_length])
            y[i] = torch.tensor(self.tokens[start+1:start+self.seq_length+1])
        
        return x.to(device), y.to(device)


def download_wikitext():
    """Download WikiText-2 dataset."""
    print("Downloading WikiText-2...")
    
    # Try to download from HuggingFace or use a simple text file
    url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
        print(f"Downloaded {len(text):,} characters")
        return text
    except Exception as e:
        print(f"Failed to download WikiText: {e}")
        print("Using fallback: generating synthetic but structured text...")
        return generate_structured_text()


def generate_structured_text():
    """Generate structured text with patterns (fallback)."""
    # Create text with clear patterns
    text = []
    
    # Pattern 1: Simple sequences
    for i in range(1000):
        text.append(f"The quick brown fox jumps over the lazy dog. ")
        text.append(f"Pack my box with five dozen liquor jugs. ")
        text.append(f"How vexingly quick daft zebras jump! ")
    
    # Pattern 2: Number sequences
    for i in range(100):
        text.append(f"Count: 0 1 2 3 4 5 6 7 8 9. ")
        text.append(f"Fibonacci: 1 1 2 3 5 8 13 21 34. ")
    
    # Pattern 3: Structured data
    for i in range(500):
        text.append(f"Name: Alice Age: 25 City: NYC. ")
        text.append(f"Name: Bob Age: 30 City: SF. ")
        text.append(f"Name: Charlie Age: 35 City: LA. ")
    
    full_text = ''.join(text)
    print(f"Generated {len(full_text):,} characters of structured text")
    return full_text


class LanguageModel(nn.Module):
    """Simple language model wrapper for NestedModelWithSurprise."""
    
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Nested model core
        self.nested = NestedModelWithSurprise(
            input_size=embed_size,
            hidden_size=hidden_size
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_size, vocab_size)
        
        # Store for access
        self.levels = self.nested.levels
    
    def forward(self, x, compute_surprise=False):
        """
        Args:
            x: (batch, seq_len) token indices
            compute_surprise: whether to track activations
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            surprise_info: dict or None
        """
        # Embed tokens
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        
        # Run through nested model
        hidden, surprise_info = self.nested(embedded, compute_surprise=compute_surprise)
        
        # Project to vocabulary
        logits = self.output_proj(hidden)  # (batch, seq_len, vocab_size)
        
        return logits, surprise_info


def run_experiment(
    name: str,
    dataset: TextDataset,
    use_surprise: bool,
    surprise_weights: dict,
    num_steps: int = 2000,
    device: torch.device = None
):
    """Run a single experiment on real data."""
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*70}")
    print(f"Use surprise: {use_surprise}")
    print(f"Surprise weights: {surprise_weights}")
    print(f"Steps: {num_steps}")
    print(f"{'='*70}\n")
    
    # Set seed
    set_seed(42)
    
    # Initialize model
    model = LanguageModel(
        vocab_size=dataset.tokenizer.vocab_size,
        embed_size=256,
        hidden_size=512
    )
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize scheduler
    chunk_sizes = {
        "level1_fast": 1,
        "level2_medium": 16,
        "level3_slow": 256,
    }
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    # Setup optimizers
    base_lr = 3e-4  # Higher LR for real learning
    optimizers = setup_optimizers(
        model=model.nested,
        chunk_sizes=chunk_sizes,
        base_lr=base_lr,
        optimizer_type="adam"
    )
    
    # Create surprise computer
    surprise_computer = SurpriseLossComputer(
        loss_weights=surprise_weights,
        gradient_clip_value=10.0,
        compute_surprise_every_n_steps=1
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Metrics
    metrics = {
        'losses': [],
        'perplexities': [],
        'aux_losses': [],
        'step_times': []
    }
    
    # Training loop
    print(f"Training for {num_steps} steps...")
    start_time = time.time()
    
    for step in range(num_steps):
        global_step = step + 1
        
        # Get batch
        x, y = dataset.get_batch(batch_size=16, device=device)
        
        # Forward pass
        step_start = time.time()
        
        if use_surprise:
            logits, surprise_info = model(x, compute_surprise=True)
        else:
            logits, _ = model(x, compute_surprise=False)
            surprise_info = None
        
        # Compute loss
        # Reshape for cross entropy: (batch*seq, vocab)
        logits_flat = logits.view(-1, dataset.tokenizer.vocab_size)
        y_flat = y.view(-1)
        main_loss = criterion(logits_flat, y_flat)
        
        # Surprise loss
        aux_loss = 0.0
        if use_surprise and surprise_info:
            aux_loss_tensor, _ = surprise_computer.compute(
                main_loss, surprise_info, force_compute=True
            )
            aux_loss = aux_loss_tensor.item()
            total_loss = main_loss + aux_loss_tensor
        else:
            total_loss = main_loss
        
        # Backward
        total_loss.backward()
        
        # Update
        for level_name, module in model.levels.items():
            if scheduler.should_update(level_name, global_step):
                optimizers[level_name].step()
                scheduler.mark_updated(level_name, global_step)
                
                for p in module.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
        
        # Also zero embedding and projection gradients after level1 updates
        if scheduler.should_update("level1_fast", global_step):
            model.embedding.zero_grad()
            model.output_proj.zero_grad()
        
        step_time = time.time() - step_start
        
        # Record metrics
        loss_val = main_loss.item()
        perplexity = torch.exp(main_loss).item()
        
        metrics['losses'].append(loss_val)
        metrics['perplexities'].append(perplexity)
        metrics['aux_losses'].append(aux_loss)
        metrics['step_times'].append(step_time)
        
        # Progress
        if (step + 1) % 200 == 0:
            avg_loss = sum(metrics['losses'][-200:]) / 200
            avg_ppl = sum(metrics['perplexities'][-200:]) / 200
            avg_aux = sum(metrics['aux_losses'][-200:]) / 200
            elapsed = time.time() - start_time
            
            print(f"  Step {global_step}/{num_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"PPL: {avg_ppl:.2f} | "
                  f"Aux: {avg_aux:.4f} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    # Final stats
    final_loss = sum(metrics['losses'][-100:]) / 100
    final_ppl = sum(metrics['perplexities'][-100:]) / 100
    initial_ppl = sum(metrics['perplexities'][:100]) / 100
    
    results = {
        'name': name,
        'use_surprise': use_surprise,
        'surprise_weights': surprise_weights,
        'num_steps': num_steps,
        'final_loss': final_loss,
        'final_perplexity': final_ppl,
        'initial_perplexity': initial_ppl,
        'ppl_improvement': initial_ppl - final_ppl,
        'total_time': total_time,
        'avg_step_time': total_time / num_steps,
        'metrics': {
            'losses': metrics['losses'][::10],  # Subsample for storage
            'perplexities': metrics['perplexities'][::10],
        }
    }
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {name}")
    print(f"{'='*70}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Final perplexity: {final_ppl:.2f}")
    print(f"PPL improvement: {initial_ppl:.2f} → {final_ppl:.2f} ({initial_ppl - final_ppl:.2f})")
    print(f"Total time: {total_time:.1f}s ({total_time/num_steps*1000:.1f}ms/step)")
    print(f"{'='*70}\n")
    
    return results


def main():
    """Run real data experiments."""
    
    print("\n" + "="*70)
    print("REAL DATA EXPERIMENTS")
    print("="*70)
    print("\nTesting surprise objectives on structured language data")
    print("This is the REAL test - will surprise help?\n")
    
    device = get_device(prefer_cuda=True)
    
    # Download data
    text = download_wikitext()
    
    # Create tokenizer and dataset
    print("\nPreparing dataset...")
    tokenizer = SimpleTokenizer(text)
    dataset = TextDataset(text, tokenizer, seq_length=64)
    
    # Run experiments
    all_results = []
    
    # Experiment 1: Baseline
    results = run_experiment(
        name="baseline",
        dataset=dataset,
        use_surprise=False,
        surprise_weights={},
        num_steps=2000,
        device=device
    )
    all_results.append(results)
    
    # Experiment 2: Low surprise
    results = run_experiment(
        name="surprise_low",
        dataset=dataset,
        use_surprise=True,
        surprise_weights={"level1_fast": 0.01, "level2_medium": 0.005},
        num_steps=2000,
        device=device
    )
    all_results.append(results)
    
    # Experiment 3: Medium surprise
    results = run_experiment(
        name="surprise_medium",
        dataset=dataset,
        use_surprise=True,
        surprise_weights={"level1_fast": 0.05, "level2_medium": 0.01},
        num_steps=2000,
        device=device
    )
    all_results.append(results)
    
    # Compare results
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Experiment':<20} {'Final PPL':<12} {'PPL Δ':<12} {'Time (s)':<10}")
    print("-"*70)
    
    baseline_ppl = all_results[0]['final_perplexity']
    
    for result in all_results:
        ppl_diff = baseline_ppl - result['final_perplexity']
        print(f"{result['name']:<20} "
              f"{result['final_perplexity']:<12.2f} "
              f"{ppl_diff:>+11.2f} "
              f"{result['total_time']:<10.1f}")
    
    print("-"*70)
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    best = min(all_results, key=lambda x: x['final_perplexity'])
    print(f"\nBest configuration: {best['name']}")
    print(f"  Final perplexity: {best['final_perplexity']:.2f}")
    print(f"  Improvement over baseline: {baseline_ppl - best['final_perplexity']:+.2f} PPL")
    
    # Check if surprise helped
    surprise_results = [r for r in all_results if r['use_surprise']]
    if surprise_results:
        best_surprise = min(surprise_results, key=lambda x: x['final_perplexity'])
        if best_surprise['final_perplexity'] < baseline_ppl:
            improvement = baseline_ppl - best_surprise['final_perplexity']
            print(f"\n✅ Surprise objectives HELPED!")
            print(f"   Best surprise config: {best_surprise['name']}")
            print(f"   Improvement: {improvement:.2f} PPL ({improvement/baseline_ppl*100:.1f}%)")
        else:
            print(f"\n⚠️ Surprise objectives did not help on this task")
            print(f"   Baseline still performs best")
    
    # Save results
    output_path = Path(__file__).parent / "real_data_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
