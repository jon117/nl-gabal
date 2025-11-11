# Nested Learning Quick Reference

**Quick guide for using the Nested Learning implementation**

---

## 🚀 Quick Start

### Tier 1: Basic CMS

```python
from src.model import NestedModel
from src.scheduler import ChunkedUpdateScheduler
from src.train import train_loop
from src.utils import setup_optimizers, get_device

# Setup
model = NestedModel(input_size=768, hidden_size=3072)
device = get_device(prefer_cuda=True)

chunk_sizes = {
    "level1_fast": 1,      # Updates every step
    "level2_medium": 16,   # Updates every 16 steps
    "level3_slow": 256     # Updates every 256 steps
}

scheduler = ChunkedUpdateScheduler(chunk_sizes)
optimizers = setup_optimizers(model, chunk_sizes, base_lr=1e-4)

# Train
train_loop(model, dataloader, criterion, optimizers, scheduler, 
           num_steps=1000, device=device)
```

### Tier 2: With Surprise Objectives

```python
from src.model_surprise import NestedModelWithSurprise
from src.surprise_loss import SurpriseLossComputer
from src.train_surprise import train_loop_with_surprise

# Setup (same as above, but with surprise model)
model = NestedModelWithSurprise(input_size=768, hidden_size=3072)
scheduler = ChunkedUpdateScheduler(chunk_sizes)
optimizers = setup_optimizers(model, chunk_sizes, base_lr=1e-4)

# Add surprise computer
surprise_computer = SurpriseLossComputer(
    loss_weights={"level1_fast": 0.3, "level2_medium": 0.1},
    gradient_clip_value=10.0
)

# Train with surprise
train_loop_with_surprise(
    model, dataloader, criterion, optimizers, scheduler,
    surprise_computer, num_steps=1000, device=device,
    use_surprise=True  # Toggle on/off
)
```

---

## 📋 Common Configurations

### Small Model (Fast Training)

```python
model = NestedModelWithSurprise(input_size=256, hidden_size=512)
chunk_sizes = {"level1_fast": 1, "level2_medium": 8, "level3_slow": 16}
surprise_weights = {"level1_fast": 0.1, "level2_medium": 0.05}
```

### Standard Model (Paper Settings)

```python
model = NestedModelWithSurprise(input_size=768, hidden_size=3072)
chunk_sizes = {"level1_fast": 1, "level2_medium": 16, "level3_slow": 256}
surprise_weights = {"level1_fast": 0.3, "level2_medium": 0.1}
```

### Large Model (Heavy Compute)

```python
model = NestedModelWithSurprise(input_size=1024, hidden_size=4096)
chunk_sizes = {"level1_fast": 1, "level2_medium": 32, "level3_slow": 512}
surprise_weights = {"level1_fast": 0.5, "level2_medium": 0.2}
```

---

## 🎛️ Key Parameters

### Chunk Sizes
Controls update frequency for each level.

```python
chunk_sizes = {
    "level1_fast": 1,      # Fast: updates every step
    "level2_medium": 16,   # Medium: updates every 16 steps
    "level3_slow": 256     # Slow: updates every 256 steps
}
```

**Guidelines:**
- Fast level: 1 (always)
- Medium level: 8-32 (tune based on task)
- Slow level: 64-512 (tune based on sequence length)

### Surprise Weights
Controls strength of auxiliary losses.

```python
loss_weights = {
    "level1_fast": 0.3,    # Higher for fast level
    "level2_medium": 0.1   # Lower for medium level
    # level3_slow: no surprise loss (it's the output)
}
```

**Guidelines:**
- Start small: 0.01-0.05
- Gradually increase: 0.1-0.3
- Fast level gets higher weight (updates more often)

### Learning Rates
Automatically scaled by 1/chunk_size.

```python
base_lr = 1e-4

# Actual rates:
# level1_fast:   1e-4 / 1   = 1e-4
# level2_medium: 1e-4 / 16  = 6.25e-6
# level3_slow:   1e-4 / 256 = 3.9e-7
```

**Guidelines:**
- base_lr: 1e-4 to 1e-3 (standard range)
- Scaling is automatic, don't worry about it

---

## 💾 Checkpointing

### Save Checkpoint

```python
from src.utils import save_checkpoint

save_checkpoint(
    model=model,
    optimizers=optimizers,
    scheduler=scheduler,
    global_step=1000,
    save_path="checkpoints/step_1000.pt"
)
```

### Load Checkpoint

```python
from src.utils import load_checkpoint

global_step = load_checkpoint(
    checkpoint_path="checkpoints/step_1000.pt",
    model=model,
    optimizers=optimizers,
    scheduler=scheduler
)

# Resume training from global_step
train_loop(..., start_step=global_step)
```

---

## 🧪 Testing & Debugging

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

### Quick Verification

```bash
python ai-notes/quick_test.py
```

### Test Surprise Objectives

```bash
python ai-notes/test_surprise.py
```

### Run Examples

```bash
# Basic CMS
python examples/simple_example.py

# With surprise
python examples/surprise_example.py
```

---

## 🔧 Performance Tuning

### Reduce Memory Usage

1. **Smaller batch size:**
   ```python
   batch_size = 16  # Instead of 32
   ```

2. **Compute surprise less frequently:**
   ```python
   compute_surprise_every_n_steps=4  # Every 4th step
   ```

3. **Remove surprise for some levels:**
   ```python
   loss_weights = {"level1_fast": 0.3}  # Only fast level
   ```

### Increase Speed

1. **Disable surprise:**
   ```python
   use_surprise=False
   ```

2. **Larger chunk sizes:**
   ```python
   chunk_sizes = {"level1_fast": 1, "level2_medium": 32, "level3_slow": 512}
   ```

3. **Reduce logging:**
   ```python
   log_interval=1000  # Log less frequently
   ```

### GPU Utilization

```python
# Check GPU usage
device = get_device(prefer_cuda=True)
print(f"Using: {torch.cuda.get_device_name(0)}")

# Move to GPU
model.to(device)
data = data.to(device)
```

---

## 🐛 Common Issues

### Issue: Out of Memory

**Solutions:**
1. Reduce batch size
2. Reduce model size (hidden_size)
3. Compute surprise less frequently
4. Disable surprise (use_surprise=False)

### Issue: Training Too Slow

**Check:**
- Is surprise enabled? (adds ~33% overhead)
- Are you using GPU?
- Is batch size too small?

### Issue: Gradients Exploding

**Solutions:**
1. Reduce gradient_clip_value:
   ```python
   gradient_clip_value=5.0  # Lower value
   ```

2. Reduce surprise weights:
   ```python
   loss_weights={"level1_fast": 0.05, "level2_medium": 0.01}
   ```

3. Reduce learning rate:
   ```python
   base_lr=1e-5  # Smaller
   ```

### Issue: Surprise Computation Fails

**Check:**
- Are activations being tracked?
- Is create_graph=True in grad computation?
- Try with error handling enabled

---

## 📊 Monitoring Training

### Key Metrics to Track

1. **Main loss** - Task performance
2. **Auxiliary loss** - Surprise prediction quality
3. **Update counts** - Verify scheduler working
4. **Gradient norms** - Check stability

### Logging

```python
# Built-in logging
train_loop_with_surprise(..., log_interval=50)

# Custom logging
if step % 100 == 0:
    stats = scheduler.get_stats(step)
    print(f"Updates: {stats['level1_fast']['update_count']}")
```

### TensorBoard (Future)

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('Loss/main', main_loss, step)
writer.add_scalar('Loss/aux', aux_loss, step)
```

---

## 🎯 Experiment Checklist

Before running experiments:

- [ ] Set random seed for reproducibility
- [ ] Verify GPU access
- [ ] Choose appropriate chunk sizes
- [ ] Start with small surprise weights
- [ ] Enable logging
- [ ] Set up checkpointing
- [ ] Plan evaluation metrics

During experiments:

- [ ] Monitor loss curves
- [ ] Check update counts
- [ ] Watch for instabilities
- [ ] Save checkpoints regularly
- [ ] Compare with baseline (no surprise)

After experiments:

- [ ] Analyze results
- [ ] Tune hyperparameters
- [ ] Run ablation studies
- [ ] Document findings

---

## 📚 Further Reading

- **TIER2_SURPRISE_OBJECTIVES.md** - Detailed Tier 2 docs
- **IMPLEMENTATION_SUMMARY.md** - Complete overview
- **Paper:** Nested Learning (HOPE)

---

**Quick Reference Version:** 1.0  
**Last Updated:** 2025-11-11
