"""
Unit tests for NestedModel.
"""

import pytest
import torch
from src.model import NestedModel


def test_model_initialization():
    """Test that the model initializes correctly."""
    model = NestedModel(input_size=768, hidden_size=3072)
    
    assert model.input_size == 768
    assert model.hidden_size == 3072
    assert len(model.levels) == 3
    assert "level1_fast" in model.levels
    assert "level2_medium" in model.levels
    assert "level3_slow" in model.levels


def test_forward_pass():
    """Test that forward pass produces correct output shape."""
    model = NestedModel(input_size=768, hidden_size=3072)
    
    batch_size = 4
    seq_length = 16
    input_size = 768
    
    x = torch.randn(batch_size, seq_length, input_size)
    output = model(x)
    
    assert output.shape == (batch_size, seq_length, input_size)


def test_parameter_counts():
    """Test parameter counting functionality."""
    model = NestedModel(input_size=768, hidden_size=3072)
    
    total_params = model.count_parameters()
    assert total_params > 0
    
    # Each level should have parameters
    for level_name in model.get_level_names():
        level_params = model.count_parameters(level_name)
        assert level_params > 0


def test_level_access():
    """Test accessing individual levels."""
    model = NestedModel(input_size=768, hidden_size=3072)
    
    level_names = model.get_level_names()
    assert len(level_names) == 3
    
    for level_name in level_names:
        params = list(model.get_level_params(level_name))
        assert len(params) > 0


def test_invalid_level_name():
    """Test that invalid level names raise errors."""
    model = NestedModel(input_size=768, hidden_size=3072)
    
    with pytest.raises(ValueError):
        list(model.get_level_params("invalid_level"))


def test_gradient_flow():
    """Test that gradients flow through all levels."""
    model = NestedModel(input_size=768, hidden_size=3072)
    
    x = torch.randn(2, 8, 768)
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    # Check that all levels have gradients
    for level_name in model.get_level_names():
        for param in model.get_level_params(level_name):
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
