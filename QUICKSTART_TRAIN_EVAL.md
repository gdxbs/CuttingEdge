# Quick Start: Training and Evaluation

## Setup

```bash
# Activate environment
source .venv/bin/activate  # Unix/Mac
# or
.venv\Scripts\activate     # Windows

# Ensure you have training data
# - images/shape/ should have pattern images
# - images/cloth/ should have cloth images
```

## Train the System

```bash
# Run hyperparameter optimization (5-10 minutes)
python -m cutting_edge.main --mode train

# This will:
# ✓ Test different parameter combinations
# ✓ Find the best configuration
# ✓ Save results to output/
```

**Expected output:**
```
=== HYPERPARAMETER OPTIMIZATION ===
Training samples: 15
Validation samples: 10

Testing configurations...
--- Config 1/4 ---
Testing: grid_size=20, rotation_angles=4_angles, allow_flipping=True, max_attempts=500
Train: util=45.2% success=75.0% time=1.23s
Val:   util=48.1% success=72.0% time=1.18s
✓ NEW BEST! Score: 54.16

... (more configs) ...

OPTIMIZATION COMPLETE
Best validation score: 56.23
Best configuration: grid_size=25, rotations=8_angles, flipping=True
```

## Evaluate Performance

```bash
# Test on held-out test set
python -m cutting_edge.main --mode eval

# This will:
# ✓ Load best configuration
# ✓ Evaluate on test samples
# ✓ Generate performance reports
```

**Expected output:**
```
=== ENHANCED EVALUATION MODE ===
Loading best configuration from output/best_config.json
Test set: 18 cloths, 2500 patterns
Evaluating on 20 test samples...

TEST SET EVALUATION RESULTS
======================================================================
Samples evaluated: 20
Average utilization: 52.3%
Average success rate: 68.5%
Average waste: 2847.2 cm²
Average processing time: 1.45s
Total patterns placed: 85/124
======================================================================
```

## View Results

### Training Results

```bash
# View best configuration
cat output/best_config.json

# View training summary
cat output/training_summary.txt

# View full history (JSON)
cat output/training_history.json
```

### Evaluation Results

```bash
# View evaluation summary
cat output/evaluation_summary.txt

# View detailed results (JSON)
cat output/evaluation_results.json
```

## Use Optimized Configuration

The system automatically uses the best configuration in all modes:

```bash
# Demo mode with optimized params
python -m cutting_edge.main --mode demo --num_patterns 5

# Fit mode with optimized params
python -m cutting_edge.main --mode fit --patterns pattern1.png pattern2.png --cloth cloth1.jpg

# All cloths mode with optimized params
python -m cutting_edge.main --mode all_cloths --max_patterns_per_cloth 10
```

## Understanding the Metrics

### Utilization
- **What**: Percentage of cloth area covered by patterns
- **Goal**: Maximize (higher is better)
- **Typical**: 40-70% depending on cloth complexity
- **Excellent**: 60%+

### Success Rate
- **What**: Percentage of patterns successfully placed
- **Goal**: Maximize (higher is better)
- **Typical**: 60-85%
- **Excellent**: 80%+

### Waste Area
- **What**: Unused cloth area in cm²
- **Goal**: Minimize (lower is better)
- **Depends on**: Cloth size and pattern count

### Processing Time
- **What**: Seconds to fit patterns on one cloth
- **Goal**: Minimize while maintaining quality
- **Typical**: 1-3 seconds
- **Fast**: <2 seconds

## Tips for Best Results

1. **More Training Data = Better Results**
   - Use diverse cloth sizes and types
   - Include various pattern shapes and sizes

2. **Adjust Search Space**
   - Edit `training_optimizer.py` to add more parameter options
   - Balance search space size vs. training time

3. **Monitor Validation Scores**
   - Look for consistent improvement
   - If all configs perform similarly, expand search space

4. **Check Per-Cloth-Type Performance**
   - Some cloth types may perform better/worse
   - Review `evaluation_summary.txt` for breakdowns

## Troubleshooting

**"No training data found"**
- Ensure images/shape/ and images/cloth/ have images
- Check that data_split.json was created

**"No best config found"**
- Run train mode first
- Check output/best_config.json exists

**Training is slow**
- Reduce batch_size: `--batch_size 10`
- Edit param_grid in training_optimizer.py to test fewer configs

**Low performance scores**
- Try different parameter combinations
- Ensure patterns are appropriate size for cloths
- Check if cloths have many defects

## Next Steps

1. **Run training** to optimize for your data
2. **Run evaluation** to measure performance
3. **Use demo/fit modes** with optimized configuration
4. **Iterate** - retrain with more data or adjusted parameters

## Advanced: Comparing Configurations

```python
import json
import pandas as pd

# Load training history
with open("output/training_history.json") as f:
    history = json.load(f)

# Create comparison table
df = pd.DataFrame([
    {
        "grid_size": h["config"]["grid_size"],
        "num_rotations": len(h["config"]["rotation_angles"]),
        "flipping": h["config"]["allow_flipping"],
        "val_util": h["val_metrics"]["utilization"],
        "val_success": h["val_metrics"]["success_rate"],
    }
    for h in history
])

print(df.sort_values("val_util", ascending=False))
```

## Summary

- **Training optimizes** heuristic parameters for your specific dataset
- **Evaluation measures** performance on unseen test data
- **Results are automatically applied** to all system modes
- **Comprehensive metrics** help you understand and improve performance

For more details, see `TRAIN_EVAL_IMPROVEMENTS.md`.
