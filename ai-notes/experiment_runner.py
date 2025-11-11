#!/usr/bin/env python3
"""
Experiment Runner: Testing Surprise-Based Objectives

This script runs systematic experiments to understand:
1. Effect of surprise objectives on training dynamics
2. Impact of different surprise weights
3. Interaction with chunk sizes
4. Convergence properties
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import json
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_surprise import NestedModelWithSurprise
from src.scheduler import ChunkedUpdateScheduler
from src.surprise_loss import SurpriseLossComputer
from src.train_surprise import train_step_with_surprise
from src.utils import setup_optimizers, set_seed, get_device, create_dummy_data


class ExperimentRunner:
    """Run and track experiments."""
    
    def __init__(self, device):
        self.device = device
        self.results = {}
    
    def run_experiment(
        self,
        name: str,
        use_surprise: bool,
        surprise_weights: dict,
        chunk_sizes: dict,
        num_steps: int = 500,
        base_lr: float = 1e-4
    ):
        """Run a single experiment configuration."""
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*70}")
        print(f"Use surprise: {use_surprise}")
        print(f"Surprise weights: {surprise_weights}")
        print(f"Chunk sizes: {chunk_sizes}")
        print(f"Steps: {num_steps}")
        print(f"{'='*70}\n")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Initialize model
        model = NestedModelWithSurprise(input_size=256, hidden_size=512)
        model.to(self.device)
        
        # Initialize scheduler
        scheduler = ChunkedUpdateScheduler(chunk_sizes)
        
        # Setup optimizers
        optimizers = setup_optimizers(
            model=model,
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
        
        # Setup loss
        criterion = nn.MSELoss()
        
        # Data generator
        def data_generator():
            return create_dummy_data(
                batch_size=16,
                seq_length=32,
                input_size=256,
                device=self.device
            )
        
        # Track metrics
        metrics = {
            'main_losses': [],
            'aux_losses': [],
            'total_losses': [],
            'aux_losses_by_level': defaultdict(list),
            'step_times': [],
            'update_counts': {},
            'final_scheduler_stats': {}
        }
        
        # Training loop
        print(f"Running {num_steps} training steps...")
        start_time = time.time()
        
        for step in range(num_steps):
            global_step = step + 1
            
            # Get batch
            data, targets = data_generator()
            
            # Training step
            step_start = time.time()
            main_loss, aux_loss, aux_losses_dict = train_step_with_surprise(
                model=model,
                data=data,
                targets=targets,
                criterion=criterion,
                optimizers=optimizers,
                scheduler=scheduler,
                surprise_computer=surprise_computer,
                global_step=global_step,
                device=self.device,
                use_surprise=use_surprise
            )
            step_time = time.time() - step_start
            
            # Record metrics
            metrics['main_losses'].append(main_loss)
            metrics['aux_losses'].append(aux_loss)
            metrics['total_losses'].append(main_loss + aux_loss)
            metrics['step_times'].append(step_time)
            
            for level_name, loss_val in aux_losses_dict.items():
                metrics['aux_losses_by_level'][level_name].append(loss_val.item())
            
            # Progress
            if (step + 1) % 100 == 0:
                avg_main = sum(metrics['main_losses'][-100:]) / 100
                avg_aux = sum(metrics['aux_losses'][-100:]) / 100
                print(f"  Step {global_step}/{num_steps}: "
                      f"main_loss={avg_main:.4f}, aux_loss={avg_aux:.4f}")
        
        total_time = time.time() - start_time
        
        # Get final scheduler stats
        final_stats = scheduler.get_stats(num_steps)
        for level_name, stats in final_stats.items():
            metrics['update_counts'][level_name] = stats['update_count']
            metrics['final_scheduler_stats'][level_name] = stats
        
        # Compute summary statistics
        metrics['summary'] = {
            'total_time': total_time,
            'avg_step_time': total_time / num_steps,
            'final_main_loss': metrics['main_losses'][-1],
            'final_aux_loss': metrics['aux_losses'][-1],
            'avg_main_loss': sum(metrics['main_losses']) / len(metrics['main_losses']),
            'avg_aux_loss': sum(metrics['aux_losses']) / len(metrics['aux_losses']),
            'main_loss_improvement': metrics['main_losses'][0] - metrics['main_losses'][-1],
        }
        
        print(f"\n{'='*70}")
        print(f"RESULTS: {name}")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.2f}s ({metrics['summary']['avg_step_time']*1000:.2f}ms/step)")
        print(f"Final main loss: {metrics['summary']['final_main_loss']:.4f}")
        print(f"Final aux loss: {metrics['summary']['final_aux_loss']:.4f}")
        print(f"Main loss improvement: {metrics['summary']['main_loss_improvement']:.4f}")
        print(f"\nUpdate counts:")
        for level_name, count in metrics['update_counts'].items():
            print(f"  {level_name}: {count}")
        print(f"{'='*70}\n")
        
        # Store results
        self.results[name] = metrics
        
        return metrics
    
    def compare_results(self):
        """Compare results across experiments."""
        
        print(f"\n{'='*70}")
        print("COMPARISON ACROSS EXPERIMENTS")
        print(f"{'='*70}\n")
        
        # Create comparison table
        print(f"{'Experiment':<30} {'Final Loss':<12} {'Improvement':<12} {'Time (s)':<10}")
        print(f"{'-'*70}")
        
        for name, metrics in self.results.items():
            summary = metrics['summary']
            print(f"{name:<30} "
                  f"{summary['final_main_loss']:<12.4f} "
                  f"{summary['main_loss_improvement']:<12.4f} "
                  f"{summary['total_time']:<10.2f}")
        
        print(f"{'-'*70}\n")
        
        # Analyze surprise effects
        baseline = None
        for name, metrics in self.results.items():
            if 'baseline' in name.lower():
                baseline = metrics
                break
        
        if baseline:
            print(f"{'='*70}")
            print("SURPRISE EFFECT ANALYSIS (vs Baseline)")
            print(f"{'='*70}\n")
            
            baseline_loss = baseline['summary']['final_main_loss']
            baseline_time = baseline['summary']['total_time']
            
            for name, metrics in self.results.items():
                if name == 'baseline':
                    continue
                
                loss_diff = baseline_loss - metrics['summary']['final_main_loss']
                time_overhead = (metrics['summary']['total_time'] / baseline_time - 1) * 100
                
                print(f"{name}:")
                print(f"  Loss difference: {loss_diff:+.4f} ({loss_diff/baseline_loss*100:+.2f}%)")
                print(f"  Time overhead: {time_overhead:+.2f}%")
                print()
    
    def save_results(self, filepath: str):
        """Save results to JSON."""
        # Convert to JSON-serializable format
        serializable_results = {}
        for name, metrics in self.results.items():
            serializable_results[name] = {
                'main_losses': metrics['main_losses'],
                'aux_losses': metrics['aux_losses'],
                'total_losses': metrics['total_losses'],
                'update_counts': metrics['update_counts'],
                'summary': metrics['summary']
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {filepath}")


def main():
    """Run experiment suite."""
    
    print("\n" + "="*70)
    print("SURPRISE OBJECTIVES EXPERIMENT SUITE")
    print("="*70)
    print("\nThis suite tests different surprise objective configurations")
    print("to understand their effect on training dynamics.\n")
    
    device = get_device(prefer_cuda=True)
    runner = ExperimentRunner(device)
    
    # Experiment configurations
    chunk_sizes = {
        "level1_fast": 1,
        "level2_medium": 8,
        "level3_slow": 16,
    }
    
    num_steps = 500
    
    # Experiment 1: Baseline (no surprise)
    runner.run_experiment(
        name="baseline",
        use_surprise=False,
        surprise_weights={},
        chunk_sizes=chunk_sizes,
        num_steps=num_steps
    )
    
    # Experiment 2: Low surprise weights
    runner.run_experiment(
        name="surprise_low",
        use_surprise=True,
        surprise_weights={"level1_fast": 0.05, "level2_medium": 0.01},
        chunk_sizes=chunk_sizes,
        num_steps=num_steps
    )
    
    # Experiment 3: Medium surprise weights
    runner.run_experiment(
        name="surprise_medium",
        use_surprise=True,
        surprise_weights={"level1_fast": 0.1, "level2_medium": 0.05},
        chunk_sizes=chunk_sizes,
        num_steps=num_steps
    )
    
    # Experiment 4: High surprise weights
    runner.run_experiment(
        name="surprise_high",
        use_surprise=True,
        surprise_weights={"level1_fast": 0.3, "level2_medium": 0.1},
        chunk_sizes=chunk_sizes,
        num_steps=num_steps
    )
    
    # Experiment 5: Only level1 surprise
    runner.run_experiment(
        name="surprise_level1_only",
        use_surprise=True,
        surprise_weights={"level1_fast": 0.3},
        chunk_sizes=chunk_sizes,
        num_steps=num_steps
    )
    
    # Compare results
    runner.compare_results()
    
    # Save results
    results_path = Path(__file__).parent / "experiment_results.json"
    runner.save_results(str(results_path))
    
    print("\n" + "="*70)
    print("EXPERIMENT SUITE COMPLETE!")
    print("="*70)
    print("\nKey Findings will be analyzed in the next step...")


if __name__ == "__main__":
    main()
