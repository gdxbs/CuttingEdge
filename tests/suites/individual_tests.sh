#!/bin/bash
echo "🔧 INDIVIDUAL MODE TESTS"
echo "========================"

echo "Running individual tests - press Enter to continue each test..."

read -p "Test 1: Demo mode (5 patterns) - Press Enter"
source .venv/bin/activate && python -m cutting_edge.main --mode demo --num_patterns 5

read -p "Test 2: Demo mode (25 patterns) - Press Enter" 
source .venv/bin/activate && python -m cutting_edge.main --mode demo --num_patterns 25

read -p "Test 3: All cloths mode (25 patterns each) - Press Enter"
source .venv/bin/activate && python -m cutting_edge.main --mode all_cloths --max_patterns_per_cloth 25

read -p "Test 4: Evaluation mode - Press Enter"
source .venv/bin/activate && python -m cutting_edge.main --mode eval

read -p "Test 5: Training mode (3 epochs) - Press Enter"
source .venv/bin/activate && python -m cutting_edge.main --mode train --epochs 3

echo "✅ All individual tests complete!"
