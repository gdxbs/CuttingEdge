#!/bin/bash
echo "⚡ QUICK VALIDATION TEST"
echo "====================="

# Just test core functionality
source .venv/bin/activate && python -m cutting_edge.main --mode demo --num_patterns 10

echo "✅ Quick test complete! Check output/fitting_result_*.png"
