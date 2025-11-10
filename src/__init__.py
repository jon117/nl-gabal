"""
Nested Learning (CMS) Implementation
Tier 1: Continuum Memory System with multi-timescale gradient accumulation
"""

__version__ = "0.1.0"

from .model import NestedModel
from .scheduler import ChunkedUpdateScheduler
from .train import train_loop, train_step, evaluate
from .utils import (
    load_config,
    setup_optimizers,
    set_seed,
    get_device,
    create_dummy_data,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    "NestedModel",
    "ChunkedUpdateScheduler",
    "train_loop",
    "train_step",
    "evaluate",
    "load_config",
    "setup_optimizers",
    "set_seed",
    "get_device",
    "create_dummy_data",
    "save_checkpoint",
    "load_checkpoint",
]
