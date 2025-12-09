#!/bin/bash
echo "🚀 STARTING COMPLETE CUTTING EDGE SYSTEM TEST"
echo "================================================"

echo "📊 Test 1: Demo Mode (5 patterns)"
source .venv/bin/activate && python -m cutting_edge.main --mode demo --num_patterns 5
echo ""

echo "📊 Test 2: Demo Mode (25 patterns)" 
source .venv/bin/activate && python -m cutting_edge.main --mode demo --num_patterns 25
echo ""

echo "📊 Test 3: All Cloths Mode (25 patterns each)"
source .venv/bin/activate && python -m cutting_edge.main --mode all_cloths --max_patterns_per_cloth 25
echo ""

echo "📊 Test 4: Evaluation Mode"
source .venv/bin/activate && python -m cutting_edge.main --mode eval
echo ""

echo "📊 Test 5: Training Mode (3 epochs)"
source .venv/bin/activate && python -m cutting_edge.main --mode train --epochs 3
echo ""

echo "✅ ALL TESTS COMPLETE!"
echo "📁 Check 'output/' directory for results:"
echo "   - fitting_result_*.png (visualization images)"
echo "   - cloth_analysis_*.png (cloth analysis)"  
echo "   - all_cloths_max_patterns_results.json (summary)"
echo "   - test_results.json (evaluation results)"
echo "   - training_history.json (training metrics)"
