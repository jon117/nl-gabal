"""
Unit tests for ChunkedUpdateScheduler.
"""

import pytest
from src.scheduler import ChunkedUpdateScheduler


def test_scheduler_initialization():
    """Test scheduler initialization."""
    chunk_sizes = {
        "level1_fast": 1,
        "level2_medium": 16,
        "level3_slow": 256,
    }
    
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    assert scheduler.chunk_sizes == chunk_sizes
    assert len(scheduler.last_update_step) == 3
    assert all(step == -1 for step in scheduler.last_update_step.values())


def test_should_update_logic():
    """Test step-aligned update logic."""
    chunk_sizes = {
        "level1": 1,
        "level2": 16,
        "level3": 256,
    }
    
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    # Level 1 should update every step
    assert scheduler.should_update("level1", 1) == True
    assert scheduler.should_update("level1", 2) == True
    assert scheduler.should_update("level1", 16) == True
    
    # Level 2 should update at multiples of 16
    assert scheduler.should_update("level2", 1) == False
    assert scheduler.should_update("level2", 15) == False
    assert scheduler.should_update("level2", 16) == True
    assert scheduler.should_update("level2", 32) == True
    assert scheduler.should_update("level2", 33) == False
    
    # Level 3 should update at multiples of 256
    assert scheduler.should_update("level3", 1) == False
    assert scheduler.should_update("level3", 255) == False
    assert scheduler.should_update("level3", 256) == True
    assert scheduler.should_update("level3", 512) == True


def test_mark_updated():
    """Test marking updates."""
    chunk_sizes = {"level1": 1, "level2": 16}
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    scheduler.mark_updated("level1", 10)
    assert scheduler.last_update_step["level1"] == 10
    assert scheduler.update_counts["level1"] == 1
    
    scheduler.mark_updated("level1", 20)
    assert scheduler.last_update_step["level1"] == 20
    assert scheduler.update_counts["level1"] == 2


def test_should_zero_grad():
    """Test gradient zeroing logic."""
    chunk_sizes = {"level1": 1, "level2": 16}
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    # Before any updates
    assert scheduler.should_zero_grad("level1", 1) == False
    
    # After marking update at step 16
    scheduler.mark_updated("level2", 16)
    assert scheduler.should_zero_grad("level2", 16) == True
    assert scheduler.should_zero_grad("level2", 17) == False


def test_get_stats():
    """Test statistics gathering."""
    chunk_sizes = {"level1": 1, "level2": 16}
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    # Simulate some updates
    for step in range(1, 33):
        if scheduler.should_update("level1", step):
            scheduler.mark_updated("level1", step)
        if scheduler.should_update("level2", step):
            scheduler.mark_updated("level2", step)
    
    stats = scheduler.get_stats(32)
    
    assert stats["level1"]["update_count"] == 32
    assert stats["level2"]["update_count"] == 2  # Updates at 16 and 32


def test_reset():
    """Test scheduler reset."""
    chunk_sizes = {"level1": 1}
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    scheduler.mark_updated("level1", 10)
    assert scheduler.update_counts["level1"] == 1
    
    scheduler.reset()
    assert scheduler.update_counts["level1"] == 0
    assert scheduler.last_update_step["level1"] == -1


def test_invalid_level_name():
    """Test handling of invalid level names."""
    chunk_sizes = {"level1": 1}
    scheduler = ChunkedUpdateScheduler(chunk_sizes)
    
    with pytest.raises(ValueError):
        scheduler.should_update("invalid_level", 1)
    
    with pytest.raises(ValueError):
        scheduler.mark_updated("invalid_level", 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
