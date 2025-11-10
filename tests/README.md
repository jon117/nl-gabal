# Tests

This directory contains unit tests for the Nested Learning implementation.

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_model.py -v
pytest tests/test_scheduler.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Structure

- `test_model.py`: Tests for NestedModel
  - Initialization
  - Forward pass
  - Parameter counting
  - Gradient flow

- `test_scheduler.py`: Tests for ChunkedUpdateScheduler
  - Step-aligned update logic
  - Gradient zeroing logic
  - Statistics tracking
  - State management
