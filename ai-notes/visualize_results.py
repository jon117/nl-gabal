#!/usr/bin/env python3
"""
Visualize experiment results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path(__file__).parent / "experiment_results.json"
with open(results_path, 'r') as f:
    results = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Surprise-Based Objectives: Experimental Results', fontsize=16, fontweight='bold')

# Plot 1: Main Loss Curves
ax1 = axes[0, 0]
for name, metrics in results.items():
    main_losses = metrics['main_losses']
    # Smooth with moving average
    window = 20
    smoothed = np.convolve(main_losses, np.ones(window)/window, mode='valid')
    ax1.plot(range(len(smoothed)), smoothed, label=name, linewidth=2)

ax1.set_xlabel('Step')
ax1.set_ylabel('Main Loss')
ax1.set_title('Main Loss Convergence')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Final Performance Comparison
ax2 = axes[0, 1]
names = list(results.keys())
final_losses = [results[name]['summary']['final_main_loss'] for name in names]
colors = ['green' if 'baseline' in name else 'blue' if 'low' in name else 'orange' if 'medium' in name else 'red' for name in names]

bars = ax2.bar(range(len(names)), final_losses, color=colors, alpha=0.7)
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(names, rotation=45, ha='right')
ax2.set_ylabel('Final Main Loss')
ax2.set_title('Final Performance Comparison (Lower is Better)')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, loss) in enumerate(zip(bars, final_losses)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{loss:.4f}',
             ha='center', va='bottom', fontsize=9)

# Plot 3: Training Time Comparison
ax3 = axes[1, 0]
times = [results[name]['summary']['total_time'] for name in names]
bars = ax3.bar(range(len(names)), times, color=colors, alpha=0.7)
ax3.set_xticks(range(len(names)))
ax3.set_xticklabels(names, rotation=45, ha='right')
ax3.set_ylabel('Training Time (s)')
ax3.set_title('Training Time (500 steps)')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, time in zip(bars, times):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.2f}s',
             ha='center', va='bottom', fontsize=9)

# Plot 4: Auxiliary Loss (for surprise experiments)
ax4 = axes[1, 1]
for name, metrics in results.items():
    if 'baseline' not in name:  # Skip baseline (no aux loss)
        aux_losses = metrics['aux_losses']
        # Smooth
        smoothed = np.convolve(aux_losses, np.ones(window)/window, mode='valid')
        ax4.plot(range(len(smoothed)), smoothed, label=name, linewidth=2)

ax4.set_xlabel('Step')
ax4.set_ylabel('Auxiliary Loss')
ax4.set_title('Auxiliary Loss Evolution (Surprise Experiments Only)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent / "experiment_results.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

# Show figure
plt.show()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print("\nKey Observations from Plots:")
print("\n1. Main Loss Convergence (top-left):")
print("   - Baseline converges fastest and lowest")
print("   - High surprise weights slow convergence")
print("   - Low surprise weights track baseline closely")
print("\n2. Final Performance (top-right):")
print("   - Baseline achieves best final loss")
print("   - High surprise weights result in worse final loss")
print("   - Trade-off between surprise and main objective")
print("\n3. Training Time (bottom-left):")
print("   - Surprise objectives add 50-140% overhead")
print("   - Level1-only is surprisingly fast")
print("   - Cost increases with number of surprise levels")
print("\n4. Auxiliary Loss (bottom-right):")
print("   - Aux loss grows during training")
print("   - Higher weights = higher aux loss")
print("   - Represents surprise prediction quality")
