# Nested Learning (CMS) Implementation

An implementation of the Nested Learning paper from Google, focusing on the Continuum Memory System (CMS) with multi-timescale gradient accumulation.

## Overview

This project implements a hierarchical learning architecture where different parameter groups update at different frequencies:
- **Fast Level** (Level 1): Updates every step for rapid adaptation
- **Medium Level** (Level 2): Accumulates gradients over 16 steps for mid-term patterns
- **Slow Level** (Level 3): Accumulates gradients over 256 steps for long-term structure

## Project Structure

```
nl-gabal/
├── src/                    # Core implementation
│   ├── model.py           # NestedModel architecture
│   ├── scheduler.py       # ChunkedUpdateScheduler
│   ├── train.py           # Training loop
│   └── utils.py           # Helper functions
├── configs/               # Configuration files
│   └── default_config.yaml
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks for demos
├── reference/             # Research papers
└── requirements.txt       # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Roadmap

- [x] **Tier 1: CMS Foundation** - Multi-timescale gradient accumulation with step-aligned updates
- [x] **Tier 2: Surprise-Based Objectives** - Auxiliary losses using second-order gradients (∇_{y_ℓ} L)
- [ ] **Tier 3: LSTM State Management** - Proper hidden state persistence
- [ ] **Tier 4: Titans Self-Referential Module** - Learning parameter update rules
- [ ] **Tier 5: GABAL** - Gated and Bounded Adaptive Learning with learned learning rates

## Key Features

- **Memory Efficient**: Uses native PyTorch gradient accumulation
- **Step-Aligned Logic**: Updates occur at precise step multiples
- **Scaled Learning Rates**: Automatically adjusts LR by 1/chunk_size
- **Surprise-Based Objectives**: Second-order gradients for auxiliary losses (Tier 2)
- **Extensible Design**: Clean separation for future enhancements
- **Toggle-able Surprise**: Easy on/off switch for ablation studies
