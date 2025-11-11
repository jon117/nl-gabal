#!/usr/bin/env python3
"""
Comprehensive Experiment: Testing All HOPE Components

Comparing:
1. Baseline (CMS only, Adam)
2. Surprise objectives (CMS + surprise, Adam)
3. Delta-rule (CMS, Delta-rule optimizer)
4. Full HOPE (CMS + surprise + Delta-rule)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import time
import json
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_surprise import NestedModelWithSurprise
from src.scheduler import ChunkedUpdateScheduler
from src.surprise_loss import SurpriseLossComputer
from src.train_surprise import train_step_with_surprise
from src.utils import setup_optimizers, set_seed, get_device
from src.delta_rule_optimizer import setup_delta_rule_optimizers


class SimpleTokenizer:
    """Character-level tokenizer."""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char.get(i, '?') for i in indices])


class TextDataset:
    """Text dataset for language modeling."""
    def __init__(self, text, tokenizer, seq_length=64):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = tokenizer.encode(text)
        
    def get_batch(self, batch_size, device):
        max_start = len(self.tokens) - self.seq_length - 1
        starts = torch.randint(0, max_start, (batch_size,))
        
        x = torch.zeros(batch_size, self.seq_length, dtype=torch.long)
        y = torch.zeros(batch_size, self.seq_length, dtype=torch.long)
        
        for i, start in enumerate(starts):
            x[i] = torch.tensor(self.tokens[start:start+self.seq_length])
            y[i] = torch.tensor(self.tokens[start+1:start+self.seq_length+1])
        
        return x.to(device), y.to(device)


class LanguageModel(nn.Module):
    """Language model with nested architecture."""
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.nested = NestedModelWithSurprise(input_size=embed_size, hidden_size=hidden_size)
        self.output_proj = nn.Linear(embed_size, vocab_size)
        self.levels = self.nested.levels
    
    def forward(self, x, compute_surprise=False):
        embedded = self.embedding(x)
        hidden, surprise_info = self.nested(embedded, compute_surprise=compute_surprise)
        logits = self.output_proj(hidden)
        return logits, surprise_info


def run_experiment(
    name: str,
    dataset: TextDataset,
    use_surprise: bool,
    use_delta_rule: bool,
    surprise_weights: dict = None,
    num_steps: int = 1000,
    device: torch.device = None
):
    """Run a single comprehensive experiment."""
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*70}")
    print(f"Use surprise: {use_surprise}")
    print(f"Use delta-rule: {use_delta_rule}")
    if surprise_weights:
        print(f"Surprise weights: {surprise_weights}")
    print(f"Steps: {num_steps}")
    print(f"{'='*70}\n")
    
    set_seed(42)
    
    # Model
    model = LanguageModel(
        vocab_size=dataset.tokenizer.vocab_size,
        embed_size=256,
        hidden_size=512
    )
    model.to(device)
    
    # Scheduler
    chunk_sizes = {
        "level1_fast": 1,
        "level2_medium": 16,
        "level3_slow": 256,
    }
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    # Optimizers
    base_lr = 3e-4
    if use_delta_rule:
        optimizers = setup_delta_rule_optimizers(
            model=model.nested,
            chunk_sizes=chunk_sizes,
            base_lr=base_lr,
            optimizer_type="delta_adam",
            decay_rate=1e-4
        )
    else:
        optimizers = setup_optimizers(
            model=model.nested,
            chunk_sizes=chunk_sizes,
            base_lr=base_lr,
            optimizer_type="adam"
        )
    
    # Surprise computer
    if use_surprise:
        surprise_computer = SurpriseLossComputer(
            loss_weights=surprise_weights,
            gradient_clip_value=10.0,
            compute_surprise_every_n_steps=1
        )
    else:
        surprise_computer = None
    
    # Training
    criterion = nn.CrossEntropyLoss()
    metrics = {'losses': [], 'perplexities': [], 'aux_losses': []}
    
    # Store inputs for delta-rule
    level_inputs = {}
    
    print(f"Training for {num_steps} steps...")
    start_time = time.time()
    
    for step in range(num_steps):
        global_step = step + 1
        
        # Get batch
        x, y = dataset.get_batch(batch_size=16, device=device)
        
        # Forward
        if use_surprise:
            logits, surprise_info = model(x, compute_surprise=True)
            # Store inputs for delta-rule
            if use_delta_rule and surprise_info:
                level_inputs = surprise_info.get('inputs', {})
        else:
            logits, _ = model(x, compute_surprise=False)
        
        # Loss
        logits_flat = logits.view(-1, dataset.tokenizer.vocab_size)
        y_flat = y.view(-1)
        main_loss = criterion(logits_flat, y_flat)
        
        # Surprise loss
        aux_loss_val = 0.0
        if use_surprise and surprise_info:
            aux_loss_tensor, _ = surprise_computer.compute(
                main_loss, surprise_info, force_compute=True
            )
            aux_loss_val = aux_loss_tensor.item()
            total_loss = main_loss + aux_loss_tensor
        else:
            total_loss = main_loss
        
        # Backward
        total_loss.backward()
        
        # Update with delta-rule inputs if available
        for level_name, module in model.levels.items():
            if scheduler.should_update(level_name, global_step):
                # Get inputs for this level
                level_input = level_inputs.get(level_name) if use_delta_rule else None
                
                # Step optimizer
                if use_delta_rule and level_input is not None:
                    optimizers[level_name].step(inputs=level_input)
                else:
                    optimizers[level_name].step()
                
                scheduler.mark_updated(level_name, global_step)
                
                # Zero gradients
                for p in module.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
        
        # Zero embedding and projection
        if scheduler.should_update("level1_fast", global_step):
            model.embedding.zero_grad()
            model.output_proj.zero_grad()
        
        # Metrics
        loss_val = main_loss.item()
        perplexity = torch.exp(main_loss).item()
        
        metrics['losses'].append(loss_val)
        metrics['perplexities'].append(perplexity)
        metrics['aux_losses'].append(aux_loss_val)
        
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
    
    # Results
    final_loss = sum(metrics['losses'][-100:]) / 100
    final_ppl = sum(metrics['perplexities'][-100:]) / 100
    initial_ppl = sum(metrics['perplexities'][:100]) / 100
    
    results = {
        'name': name,
        'use_surprise': use_surprise,
        'use_delta_rule': use_delta_rule,
        'final_loss': final_loss,
        'final_perplexity': final_ppl,
        'initial_perplexity': initial_ppl,
        'ppl_improvement': initial_ppl - final_ppl,
        'total_time': total_time,
        'metrics': {
            'losses': metrics['losses'][::10],
            'perplexities': metrics['perplexities'][::10],
        }
    }
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {name}")
    print(f"{'='*70}")
    print(f"Final perplexity: {final_ppl:.2f}")
    print(f"PPL improvement: {initial_ppl:.2f} → {final_ppl:.2f} ({initial_ppl - final_ppl:.2f})")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*70}\n")
    
    return results


def main():
    """Run comprehensive experiments."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE HOPE EXPERIMENTS")
    print("="*70)
    print("\nTesting all components:")
    print("1. Baseline (CMS + Adam)")
    print("2. Surprise (CMS + Surprise + Adam)")  
    print("3. Delta-rule (CMS + Delta-rule)")
    print("4. Full HOPE (CMS + Surprise + Delta-rule)")
    print("="*70)
    
    device = get_device(prefer_cuda=True)
    
    # Download data
    print("\nDownloading WikiText-2...")
    url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    try:
        response = requests.get(url, timeout=30)
        text = response.text
        print(f"Downloaded {len(text):,} characters")
    except:
        print("Download failed, using synthetic data...")
        text = "The quick brown fox jumps over the lazy dog. " * 10000
    
    # Prepare dataset
    tokenizer = SimpleTokenizer(text)
    dataset = TextDataset(text, tokenizer, seq_length=64)
    print(f"Vocabulary: {tokenizer.vocab_size} characters")
    print(f"Dataset: {len(dataset.tokens):,} tokens\n")
    
    # Run experiments
    all_results = []
    num_steps = 1500
    
    # 1. Baseline
    results = run_experiment(
        name="1_baseline_cms_adam",
        dataset=dataset,
        use_surprise=False,
        use_delta_rule=False,
        num_steps=num_steps,
        device=device
    )
    all_results.append(results)
    
    # 2. Surprise + Adam
    results = run_experiment(
        name="2_surprise_adam",
        dataset=dataset,
        use_surprise=True,
        use_delta_rule=False,
        surprise_weights={"level1_fast": 0.05, "level2_medium": 0.01},
        num_steps=num_steps,
        device=device
    )
    all_results.append(results)
    
    # 3. Delta-rule only
    results = run_experiment(
        name="3_delta_rule_only",
        dataset=dataset,
        use_surprise=False,
        use_delta_rule=True,
        num_steps=num_steps,
        device=device
    )
    all_results.append(results)
    
    # 4. Full HOPE
    results = run_experiment(
        name="4_full_hope",
        dataset=dataset,
        use_surprise=True,
        use_delta_rule=True,
        surprise_weights={"level1_fast": 0.05, "level2_medium": 0.01},
        num_steps=num_steps,
        device=device
    )
    all_results.append(results)
    
    # Compare
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Configuration':<25} {'Final PPL':<12} {'PPL Δ':<12} {'Time (s)':<10}")
    print("-"*70)
    
    baseline_ppl = all_results[0]['final_perplexity']
    
    for result in all_results:
        ppl_diff = baseline_ppl - result['final_perplexity']
        print(f"{result['name']:<25} "
              f"{result['final_perplexity']:<12.2f} "
              f"{ppl_diff:>+11.2f} "
              f"{result['total_time']:<10.1f}")
    
    print("-"*70)
    
    # Find best
    best = min(all_results, key=lambda x: x['final_perplexity'])
    print(f"\n🏆 Best configuration: {best['name']}")
    print(f"   Final PPL: {best['final_perplexity']:.2f}")
    print(f"   Improvement: {baseline_ppl - best['final_perplexity']:+.2f} PPL")
    
    # Save
    output_path = Path(__file__).parent / "comprehensive_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
