#!/bin/bash
# Quick test script to verify installation

echo "Testing Nested Learning (CMS) Implementation"
echo "=============================================="
echo ""

echo "1. Running unit tests..."
python -m pytest tests/ -v --tb=short

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All tests passed!"
    echo ""
    echo "2. Quick functionality check..."
    python -c "
import sys
sys.path.insert(0, 'src')
from model import NestedModel
from scheduler import ChunkedUpdateScheduler

# Initialize
model = NestedModel()
scheduler = ChunkedUpdateScheduler({
    'level1_fast': 1,
    'level2_medium': 16,
    'level3_slow': 256
})

# Test basic functionality
print('✓ Model initialized with', model.count_parameters(), 'parameters')
print('✓ Scheduler initialized with 3 levels')

# Test update logic
assert scheduler.should_update('level1_fast', 1) == True
assert scheduler.should_update('level2_medium', 16) == True
assert scheduler.should_update('level3_slow', 256) == True
print('✓ Update logic working correctly')

print('')
print('All systems operational! 🚀')
"
else
    echo ""
    echo "✗ Tests failed. Please check the errors above."
    exit 1
fi
