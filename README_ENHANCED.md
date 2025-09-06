# Cutting Edge: Enhanced Pattern Fitting System

## Overview

This is the enhanced version of the Cutting Edge system that combines the **simplicity** of our beginner-friendly approach with **advanced features** from the original system. It provides optimal pattern fitting on cloth materials while remaining easy to understand and modify.

## Key Improvements Over Simple System

### 1. **Pattern Recognition**
- **Pattern Type Classification**: Automatically identifies shirt, pants, dress, collar, etc.
- **Advanced Contour Extraction**: Better shape detection using adaptive thresholding
- **Standard Pattern Templates**: Built-in dimensions for common pattern types
- **Confidence Scores**: Shows how certain the system is about pattern classification

### 2. **Cloth Recognition**
- **Irregular Cloth Shapes**: Handles non-rectangular cloth materials
- **Defect Detection**: Identifies holes, stains, or unusable areas in cloth
- **Material Type Detection**: Classifies cotton, silk, wool, polyester, etc.
- **Texture Analysis**: Considers fabric grain and material properties
- **U-Net Segmentation**: Option for pixel-perfect cloth boundary detection

### 3. **Pattern Fitting**
- **Multiple Orientations**: Tries rotations (0°, 90°, 180°, 270°) and flipping
- **Smart Placement**: Neural network suggests optimal positions
- **Polygon-Based Collision**: Accurate overlap detection for irregular shapes
- **Edge Placement Bonus**: Rewards efficient edge utilization
- **Defect Avoidance**: Automatically avoids placing patterns over defects

### 4. **System Features**
- **Centralized Configuration**: All settings in `simple_config.py`
- **ImageNet Normalization**: Better model performance
- **Learning Rate Scheduling**: Improved training convergence
- **Checkpoint Support**: Save and resume training
- **Comprehensive Reports**: Detailed placement information

## Quick Start

### 1. Installation

```bash
# Activate environment
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install dependencies (same as before)
uv pip install -e ".[dev]"
```

### 2. Run Demo

```bash
# Run demonstration with automatic pattern selection
python -m src.cutting_edge.enhanced_main --mode demo --num_patterns 3

# This will:
# 1. Scan for pattern and cloth images
# 2. Automatically split data into train/test sets
# 3. Select patterns and cloth from test set
# 4. Perform fitting with multiple orientations
# 5. Generate visualizations and reports
```

### 3. Fit Specific Patterns

```bash
# Fit specific patterns on specific cloth
python -m src.cutting_edge.enhanced_main --mode fit \
    --patterns images/shape/pattern_50x80.png images/shape/pattern_30x40.png \
    --cloth images/cloth/cloth_200x300.jpeg
```

### 4. Train Models

```bash
# Train the enhanced models
python -m src.cutting_edge.enhanced_main --mode train
```

## Configuration

All settings are in `src/cutting_edge/simple_config.py`:

```python
# Pattern Recognition
PATTERN = {
    "DEFAULT_WIDTH": 100,       # Default pattern width in cm
    "DEFAULT_HEIGHT": 150,      # Default pattern height in cm
    "TYPES": ["shirt", "pants", "dress", "sleeve", "collar", "other"],
    "PIXEL_TO_CM": 0.1,        # Conversion factor
    ...
}

# Cloth Recognition
CLOTH = {
    "DEFAULT_WIDTH": 200,       # Default cloth width in cm
    "DEFAULT_HEIGHT": 300,      # Default cloth height in cm
    "EDGE_MARGIN": 5,          # Safety margin from edges
    "TYPES": ["cotton", "silk", "wool", "polyester", "mixed", "other"],
    ...
}

# Pattern Fitting
FITTING = {
    "ROTATION_ANGLES": [0, 90, 180, 270],  # Angles to try
    "REWARDS": {
        "overlap_penalty": -10,
        "edge_bonus": 2,
        "utilization_bonus": 5,
        "compactness_bonus": 3,
        "gap_penalty": -1,
    },
    ...
}
```

## Understanding the Enhanced Architecture

### Pattern Processing Pipeline

```
Pattern Image → Pattern Recognizer → Pattern Object
                        ↓                    ↓
                 [Type Classification]  [Shape & Dimensions]
                        ↓                    ↓
                 shirt/pants/etc.    width×height + contour
```

### Cloth Processing Pipeline

```
Cloth Image → Cloth Segmenter → Cloth Material Object
                     ↓                      ↓
             [Shape Detection]      [Defect Detection]
                     ↓                      ↓
            Irregular boundary      Holes/unusable areas
```

### Fitting Pipeline

```
Patterns + Cloth → Placement Optimizer → Fitted Layout
                           ↓                    ↓
                  [Try Orientations]    [Calculate Score]
                           ↓                    ↓
                   Rotation + Flip      Edge/Compact bonus
```

## Output Files

### 1. **Enhanced Fitting Visualization**
- `output/enhanced_fitting_*.png`
- Shows cloth shape with defects
- Placed patterns with labels
- Failed patterns shown on side
- Material utilization metrics

### 2. **Cloth Analysis Visualization**
- `output/cloth_analysis_*.png`
- Shows cloth boundary detection
- Highlights defects in red
- Displays material properties

### 3. **Detailed Report**
- `output/enhanced_fitting_*_report.txt`
- Complete placement details
- Position, rotation, and flip status
- Individual placement scores
- Material efficiency metrics

### 4. **JSON Results**
- `output/enhanced_fitting_results.json`
- Machine-readable results
- Historical tracking
- Performance metrics

## Example Report

```
ENHANCED PATTERN FITTING REPORT
============================================================

Generated: 2025-09-06 15:46:01

CLOTH INFORMATION
------------------------------------------------------------
Type: wool
Dimensions: 200.0 x 300.0 cm
Usable Area: 59,500.0 cm²
Defects: 2
Material Efficiency: 99.2%

FITTING SUMMARY
------------------------------------------------------------
Total Patterns: 5
Patterns Placed: 4
Success Rate: 80.0%
Material Utilization: 15.3%
Total Pattern Area: 9,100.0 cm²
Waste Area: 50,400.0 cm²

PLACEMENT DETAILS
------------------------------------------------------------

Pattern 1: shirt_front
  Type: shirt
  Size: 60.0 x 80.0 cm
  Area: 4,800.0 cm²
  Position: (0.0, 0.0) cm
  Rotation: 0°
  Flipped: No
  Placement Score: 85.2

Pattern 2: shirt_back
  Type: shirt
  Size: 60.0 x 80.0 cm
  Area: 4,800.0 cm²
  Position: (60.0, 0.0) cm
  Rotation: 0°
  Flipped: Yes
  Placement Score: 82.1
...
```

## Advanced Features

### 1. **Multi-Pass Fitting**
The system tries multiple strategies:
- First pass: Neural network suggestions
- Second pass: Grid-based search
- Third pass: Random sampling for remaining space

### 2. **Defect-Aware Placement**
- Automatically detects cloth defects
- Avoids placing patterns over unusable areas
- Visualizes defects in output

### 3. **Material-Specific Optimization**
- Different strategies for different cloth types
- Considers fabric grain direction
- Adapts to material stretchiness

### 4. **Batch Processing**
```bash
# Process multiple cloth-pattern combinations
for cloth in images/cloth/*.jpeg; do
    python -m src.cutting_edge.enhanced_main --mode fit \
        --patterns images/shape/*.png \
        --cloth "$cloth"
done
```

## Comparison: Simple vs Enhanced

| Feature | Simple System | Enhanced System |
|---------|--------------|-----------------|
| **Pattern Types** | Generic | Classified (shirt, pants, etc.) |
| **Cloth Shapes** | Rectangular only | Any shape + defects |
| **Orientations** | Basic rotation | Full rotation + flip |
| **Placement** | Grid search | Neural network + grid |
| **Visualization** | Basic | Detailed with defects |
| **Configuration** | In-code | Centralized config file |
| **Reports** | Basic | Comprehensive |

## When to Use Which System?

### Use **Simple System** when:
- Learning computer vision/RL basics
- Working with clean, rectangular cloth
- Need quick prototyping
- Want minimal complexity

### Use **Enhanced System** when:
- Working with real-world materials
- Need optimal material utilization
- Handling irregular shapes/defects
- Want production-ready features

## Troubleshooting

1. **Low utilization with small patterns**: The system detected very small pattern areas. Check if pattern images are being processed correctly.

2. **Pattern type misclassification**: The neural network may need more training data for accurate classification.

3. **Defect over-detection**: Adjust `HSV_LOWER` and `HSV_UPPER` in config for your cloth colors.

4. **Slow fitting with many patterns**: Reduce `GRID_SAMPLE_SIZE` for faster (but potentially less optimal) results.

## Development

### Adding New Pattern Types

1. Add to `PATTERN["TYPES"]` in config
2. Add standard dimensions to `STANDARD_PATTERNS` dict
3. Retrain pattern recognizer

### Customizing Rewards

Modify `FITTING["REWARDS"]` in config:
```python
"REWARDS": {
    "overlap_penalty": -10,    # Increase for stricter overlap prevention
    "edge_bonus": 2,          # Increase to prefer edge placement
    "utilization_bonus": 5,   # Increase to pack more tightly
    "compactness_bonus": 3,   # Increase to keep patterns together
    "gap_penalty": -1,        # Increase to avoid small gaps
}
```

## Future Enhancements

1. **3D Visualization**: Show how patterns would look on actual garments
2. **Multi-Layer Support**: Handle patterns that need multiple fabric layers
3. **Grain Direction**: Ensure patterns align with fabric grain
4. **Waste Optimization**: Suggest how to use leftover pieces
5. **Pattern Nesting**: Advanced nesting algorithms for complex shapes

## Conclusion

The Enhanced Cutting Edge system provides a production-ready solution for pattern fitting while maintaining the clarity and simplicity that makes it suitable for learning. It demonstrates how to gradually add complexity to a simple system while keeping it maintainable and understandable.