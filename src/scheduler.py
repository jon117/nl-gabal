"""
ChunkedUpdateScheduler: Orchestrates multi-timescale gradient accumulation.

This scheduler manages when each learning level should update its parameters,
using step-aligned logic for memory-efficient gradient accumulation.

Key Principles:
- Step-aligned updates: Updates occur at specific step multiples (16, 32, 48...)
  NOT after N steps have passed
- Selective gradient zeroing: Only zero gradients for levels that just updated
- Memory efficient: Relies on PyTorch's native gradient accumulation
"""

from typing import Dict


class ChunkedUpdateScheduler:
    """
    Manages updates using step-aligned, memory-efficient gradient accumulation.
    
    This scheduler determines when each learning level should:
    1. Apply gradients and update parameters (via optimizer.step())
    2. Zero gradients to start fresh accumulation
    
    Design:
        - Tracks last_update_step for each level to manage state
        - should_update(): Returns True if global_step is a multiple of chunk_size
        - should_zero_grad(): Returns True only for levels that just updated
    
    Args:
        chunk_sizes: Dictionary mapping level names to their update frequencies
                    Example: {"level1_fast": 1, "level2_medium": 16, "level3_slow": 256}
    
    Example:
        >>> scheduler = ChunkedUpdateScheduler({
        ...     "level1_fast": 1,
        ...     "level2_medium": 16,
        ...     "level3_slow": 256
        ... })
        >>> 
        >>> # At step 16:
        >>> scheduler.should_update("level1_fast", 16)  # True (updates every step)
        >>> scheduler.should_update("level2_medium", 16)  # True (16 % 16 == 0)
        >>> scheduler.should_update("level3_slow", 16)  # False (16 % 256 != 0)
    """
    
    def __init__(self, chunk_sizes: Dict[str, int]):
        """
        Initialize the scheduler with chunk sizes for each level.
        
        Args:
            chunk_sizes: Dict mapping level names to update frequencies (chunk sizes)
        """
        self.chunk_sizes = chunk_sizes
        
        # Track the last update step for each level to manage state
        # Initialized to -1 to indicate no updates have occurred yet
        self.last_update_step: Dict[str, int] = {
            level: -1 for level in chunk_sizes.keys()
        }
        
        # Statistics tracking
        self.update_counts: Dict[str, int] = {
            level: 0 for level in chunk_sizes.keys()
        }
        
        print(f"ChunkedUpdateScheduler initialized with chunk sizes: {self.chunk_sizes}")
    
    def should_update(self, level_name: str, global_step: int) -> bool:
        """
        Determines if a level should update at the current step (step-aligned).
        
        Updates occur at specific step multiples:
        - level1_fast (chunk_size=1): steps 1, 2, 3, 4, ...
        - level2_medium (chunk_size=16): steps 16, 32, 48, 64, ...
        - level3_slow (chunk_size=256): steps 256, 512, 768, ...
        
        Args:
            level_name: Name of the level to check
            global_step: Current training step (1-indexed)
        
        Returns:
            True if this level should perform an optimizer step at this global_step
        """
        if level_name not in self.chunk_sizes:
            raise ValueError(f"Unknown level: {level_name}. Available levels: {list(self.chunk_sizes.keys())}")
        
        chunk_size = self.chunk_sizes[level_name]
        
        # Update at steps that are multiples of chunk_size
        # Note: global_step > 0 ensures we don't update at step 0
        is_update_step = (global_step > 0) and (global_step % chunk_size == 0)
        
        return is_update_step
    
    def mark_updated(self, level_name: str, global_step: int):
        """
        Records that a level was updated at the given step.
        
        This is used to track state for should_zero_grad() and statistics.
        
        Args:
            level_name: Name of the level that was updated
            global_step: Step at which the update occurred
        """
        if level_name not in self.chunk_sizes:
            raise ValueError(f"Unknown level: {level_name}")
        
        self.last_update_step[level_name] = global_step
        self.update_counts[level_name] += 1
    
    def should_zero_grad(self, level_name: str, global_step: int) -> bool:
        """
        Determines if a level's gradients should be zeroed at the current step.
        
        Gradients should be zeroed ONLY for levels that just updated in this step.
        This enables selective accumulation - levels that didn't update keep their
        gradients to continue accumulating.
        
        Args:
            level_name: Name of the level to check
            global_step: Current training step
        
        Returns:
            True if this level's gradients should be zeroed (i.e., it just updated)
        """
        if level_name not in self.last_update_step:
            raise ValueError(f"Unknown level: {level_name}")
        
        # Zero gradients only if this level was just updated at this step
        return self.last_update_step[level_name] == global_step
    
    def get_update_count(self, level_name: str) -> int:
        """
        Get the total number of updates performed for a level.
        
        Args:
            level_name: Name of the level
        
        Returns:
            Number of updates performed
        """
        if level_name not in self.update_counts:
            raise ValueError(f"Unknown level: {level_name}")
        return self.update_counts[level_name]
    
    def get_stats(self, global_step: int) -> Dict[str, Dict]:
        """
        Get statistics about updates for all levels.
        
        Args:
            global_step: Current training step
        
        Returns:
            Dictionary with statistics for each level
        """
        stats = {}
        for level_name in self.chunk_sizes.keys():
            chunk_size = self.chunk_sizes[level_name]
            update_count = self.update_counts[level_name]
            last_update = self.last_update_step[level_name]
            
            # Calculate expected number of updates based on step-aligned logic
            expected_updates = global_step // chunk_size
            
            stats[level_name] = {
                "chunk_size": chunk_size,
                "update_count": update_count,
                "expected_updates": expected_updates,
                "last_update_step": last_update,
                "steps_since_update": global_step - last_update if last_update >= 0 else global_step,
            }
        
        return stats
    
    def print_stats(self, global_step: int):
        """
        Print formatted statistics about scheduler state.
        
        Args:
            global_step: Current training step
        """
        print(f"\n{'='*70}")
        print(f"ChunkedUpdateScheduler Statistics (Step {global_step})")
        print(f"{'='*70}")
        print(f"{'Level':<20} {'Chunk Size':<12} {'Updates':<10} {'Last Update':<12} {'Pending':<10}")
        print(f"{'-'*70}")
        
        stats = self.get_stats(global_step)
        for level_name, level_stats in stats.items():
            pending = level_stats['steps_since_update']
            print(
                f"{level_name:<20} "
                f"{level_stats['chunk_size']:<12} "
                f"{level_stats['update_count']:<10} "
                f"{level_stats['last_update_step']:<12} "
                f"{pending:<10}"
            )
        print(f"{'='*70}\n")
    
    def reset(self):
        """Reset the scheduler state (useful for starting new training runs)."""
        self.last_update_step = {level: -1 for level in self.chunk_sizes.keys()}
        self.update_counts = {level: 0 for level in self.chunk_sizes.keys()}
        print("ChunkedUpdateScheduler state reset.")
