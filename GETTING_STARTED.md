# Getting Started with Nested Learning (CMS)

Welcome! This guide will help you get started with the Continuum Memory System implementation.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Run all tests
./test.sh

# Or manually:
pytest tests/ -v
```

### 3. Try the Demo

**Option A: Jupyter Notebook (Recommended)**
```bash
jupyter notebook notebooks/01_cms_demo.ipynb
```

**Option B: Python Script**
```bash
python examples/simple_example.py
```

## Basic Usage

### Minimal Example

```python
import torch
import torch.nn as nn
from src.model import NestedModel
from src.scheduler import ChunkedUpdateScheduler
from src.utils import setup_optimizers, set_seed

# 1. Initialize
set_seed(42)
model = NestedModel(input_size=768, hidden_size=3072)

# 2. Setup Scheduler
chunk_sizes = {
    "level1_fast": 1,      # Updates every step
    "level2_medium": 16,   # Updates every 16 steps
    "level3_slow": 256,    # Updates every 256 steps
}
scheduler = ChunkedUpdateScheduler(chunk_sizes)

# 3. Setup Optimizers (with scaled learning rates)
optimizers = setup_optimizers(
    model=model,
    chunk_sizes=chunk_sizes,
    base_lr=1e-4
)

# 4. Training Loop
criterion = nn.MSELoss()
global_step = 0

for data, targets in dataloader:  # Your dataloader
    global_step += 1
    
    # Forward & Backward
    output = model(data)
    loss = criterion(output, targets)
    loss.backward()
    
    # Selective Update & Gradient Zeroing
    for level_name, module in model.levels.items():
        if scheduler.should_update(level_name, global_step):
            optimizers[level_name].step()
            scheduler.mark_updated(level_name, global_step)
            
            # Zero gradients only for this level
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.zero_()
```

## Understanding the Implementation

### Core Concepts

1. **Multi-Timescale Learning**: Different parameters update at different frequencies
   - Fast parameters adapt quickly to recent patterns
   - Slow parameters capture long-term structure

2. **Step-Aligned Updates**: Updates occur at specific step multiples
   - Level 2 (medium): steps 16, 32, 48, 64...
   - Level 3 (slow): steps 256, 512, 768...

3. **Learning Rate Scaling**: LR is scaled by 1/chunk_size
   - Compensates for gradient accumulation
   - Maintains effective step size across levels

4. **Memory Efficiency**: Uses native PyTorch gradient accumulation
   - No additional memory overhead
   - Single forward/backward pass

### Architecture Overview

```
NestedModel
├── Level 1 (Fast): LSTM
│   └── Updates: Every step (chunk_size=1)
├── Level 2 (Medium): Feed-forward
│   └── Updates: Every 16 steps (chunk_size=16)
└── Level 3 (Slow): Feed-forward
    └── Updates: Every 256 steps (chunk_size=256)
```

## Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
model:
  input_size: 768
  hidden_size: 3072

scheduler:
  chunk_sizes:
    level1_fast: 1
    level2_medium: 16
    level3_slow: 256

optimizer:
  base_lr: 0.0001
  optimizer_type: "adam"
```

Load config:
```python
from src.utils import load_config

config = load_config('configs/default_config.yaml')
```

## Next Steps

### Explore
- 📓 **Jupyter Demo**: `notebooks/01_cms_demo.ipynb` - Interactive walkthrough with visualizations
- 🧪 **Tests**: `tests/` - See how each component works
- 📝 **Examples**: `examples/simple_example.py` - Full training example

### Extend (Future Tiers)
- **Tier 2**: Add auxiliary losses on intermediate activations
- **Tier 3**: Implement GABAL (learned learning rates)

### Customize
- Modify chunk sizes for different timescales
- Add more or fewer levels
- Change architecture (e.g., add transformers, residual connections)

## Troubleshooting

**Import Errors**
- Make sure you're in the project root directory
- Check that `src/` is in your Python path

**Memory Issues**
- Reduce `batch_size` in your dataloader
- Use smaller `hidden_size` in model initialization

**Gradient Issues**
- Verify `scheduler.should_update()` logic
- Check that gradients are being zeroed selectively
- Ensure learning rates are properly scaled

## Resources

- 📄 **Paper**: See `reference/NL.pdf`
- 📚 **Implementation Guide**: See project README
- 💬 **Issues**: Report bugs on GitHub

## Questions?

The code is heavily commented. Start by reading:
1. `src/model.py` - The model architecture
2. `src/scheduler.py` - The update logic
3. `notebooks/01_cms_demo.ipynb` - Visual walkthrough

Happy learning! 🚀
