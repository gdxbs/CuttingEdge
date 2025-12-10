# Cutting Edge Codebase Overview

## Introduction

Cutting Edge is a sophisticated computer vision system designed to optimize garment pattern placement on cloth materials, minimizing fabric waste in the cutting process. The system leverages modern computer vision techniques and geometric analysis to achieve efficient pattern fitting while remaining accessible and understandable.

## System Architecture

The system follows a modular design with three primary components that work sequentially:

1. **Pattern Recognition Module**: Analyzes pattern images to extract shapes, dimensions, and type information
2. **Cloth Recognition Module**: Processes cloth images to identify usable areas and defects
3. **Pattern Fitting Module**: Optimizes pattern placement on the cloth using geometric operations

### Core Components

```
cutting_edge/
├── __init__.py
├── main.py                   # Main entry point and orchestration
├── config.py                 # Centralized configuration
├── pattern_recognition_module.py   # Pattern analysis
├── cloth_recognition_module.py     # Cloth material analysis
├── pattern_fitting_module.py       # Pattern placement optimization
└── training_optimizer.py           # Hyperparameter optimization (Grid Search/Bayesian)
```

## Workflow

1. **Input**: Pattern images and cloth material images
2. **Pattern Recognition**: Extract pattern contours, dimensions, and types
3. **Cloth Analysis**: Identify cloth boundaries, defects, and properties
4. **Pattern Fitting**: Optimize pattern placement for minimum waste
5. **Visualization**: Generate visual output and metrics reports

## Module Details

### Pattern Recognition Module

The Pattern Recognition Module (`pattern_recognition_module.py`) is responsible for analyzing pattern images to extract essential information needed for cutting.

#### Key Components:

- `Pattern` (dataclass): Stores pattern attributes (dimensions, contour, type)
- `PatternRecognizer` (Neural Network): Classifies pattern types and estimates dimensions
- Extraction methods: Shape detection, contour simplification, key point identification

#### Functionality:

1. **Shape Extraction**: Uses adaptive thresholding and contour detection to identify pattern shapes
2. **Pattern Classification**: Uses a CNN backbone (configurable as ResNet18 or others) to classify pattern types
3. **Dimension Estimation**: Calculates pattern dimensions through filename parsing or neural estimation
4. **Feature Extraction**: Identifies key points and simplified contours for geometric operations

### Cloth Recognition Module

The Cloth Recognition Module (`cloth_recognition_module.py`) analyzes cloth material images to prepare them for pattern fitting.

#### Key Components:

- `ClothMaterial` (dataclass): Stores cloth attributes (dimensions, contour, defects)
- `UNetSegmenter` (Neural Network): Segments cloth and defects (optional)
- Analysis methods: Color-based segmentation, defect detection, material property extraction

#### Functionality:

1. **Cloth Segmentation**: Identifies cloth boundaries using color thresholding or U-Net
2. **Defect Detection**: Identifies and classifies defects (holes, stains, lines, freeform)
3. **Cloth Type Classification**: Heuristically determines material type (leather, denim, cotton) based on shape and color measurements
4. **Material Analysis**: Extracts properties like grain direction using Gabor filters
4. **Area Calculation**: Computes total and usable area accounting for defects and margins

### Pattern Fitting Module

The Pattern Fitting Module (`pattern_fitting_module.py`) handles the core optimization of pattern placement on cloth.

#### Key Components:

- `PlacementResult` (dataclass): Stores placement information (position, rotation, etc.)
- `PlacementOptimizer` (Neural Network): Suggests optimal placements
- Geometric operations: Uses Shapely library for precise polygon manipulations

#### Functionality:

1. **Polygon Conversion**: Converts pattern and cloth contours to Shapely polygons
2. **Placement Search**: Tests various positions, rotations, and orientations 
3. **Validity Checking**: Ensures patterns don't overlap, are within boundaries, and avoid defects
5. **Auto-Scaling**: Dynamically adjusts pattern scale to maximize material utilization
6. **Dense Fallback**: Triggers a high-resolution search if standard heuristic placement fails
7. **Scoring System**: Evaluates placement quality based on edge utilization, compactness, etc.
8. **Neural Optimization**: Can use a trained network to guide placement search (optional)
9. **Visualization**: Generates detailed visualizations and reports

## Main Orchestration

The `main.py` file serves as the central orchestration point, coordinating the workflow between modules and handling user interactions.

#### Key Components:

- `CuttingEdgeSystem`: Main class that manages the entire workflow
- Command-line interface: Supports different operational modes (demo, fit, train)

#### Operation Modes:

1. **Demo Mode**: Automatically selects test images and runs the fitting process
2. **Fit Mode**: Performs fitting on user-specified pattern and cloth images
3. **Train Mode**: Trains the neural network models (placeholder implementation)

## Neural Network Architecture

The system employs several neural networks for different purposes:

1. **Pattern Recognition**: Uses a configurable backbone (ResNet18 by default) for pattern classification and dimension estimation
2. **Cloth Segmentation**: Optional U-Net architecture for precise cloth and defect segmentation
3. **Placement Optimization**: Reinforcement learning agent for suggesting optimal placements

## Geometric Optimization

The pattern fitting uses a combination of approaches:

1. **Grid Search**: Systematically tests various positions, rotations, and orientations
2. **Neural Guidance**: Optional neural network suggests promising placements
3. **Scoring Heuristics**: Edge utilization, compactness, gap penalties, and utilization bonuses
4. **Shapely Operations**: Precise geometric calculations for validity checking and transformations

## Configuration System

All parameters are centralized in `config.py` for easy tuning and experimentation:

- **System Settings**: Paths, directories, logging configuration
- **Pattern Recognition**: Image processing parameters, neural network architecture
- **Cloth Recognition**: Segmentation parameters, material detection thresholds
- **Pattern Fitting**: Placement optimization settings, reward system, rotation angles
- **Visualization**: Output display configuration
- **Training**: Batch sizes, learning rates, epochs

## Outputs and Visualization

The system generates several outputs:

1. **Fitted Layout**: Visual representation of pattern placement on cloth
2. **Cloth Analysis**: Visualization of cloth boundaries and defects
3. **Metrics**: Utilization percentage, waste area, success rate
4. **Detailed Reports**: Text reports with comprehensive placement information
5. **JSON Results**: Structured data for further analysis or integration

## Usage Examples

### Demo Mode
```bash
python -m cutting_edge.main --mode demo --num_patterns 3
```

### Fitting Specific Files
```bash
python -m cutting_edge.main --mode fit \
    --patterns images/shape/pattern_50x80.png images/shape/pattern_30x40.png \
    --cloth images/cloth/cloth_200x300.jpeg
```

### Training Mode
```bash
python -m cutting_edge.main --mode train
```

## Extending the System

The modular design makes it easy to extend the system:

1. **New Pattern Types**: Add new pattern types to `PATTERN["TYPES"]` in config.py
2. **Advanced Models**: Replace the neural network backbones with more sophisticated architectures
3. **Custom Heuristics**: Modify the placement scoring system in `calculate_placement_score()`
4. **Additional Material Properties**: Extend the `ClothMaterial` dataclass and analysis methods

## Performance Considerations

- **Computational Efficiency**: The system balances accuracy with performance
- **Neural Optimization**: Can be enabled/disabled based on computational resources
- **Configurable Complexity**: Parameters like grid size and rotation angles can be adjusted
- **Sample Limiting**: Large search spaces are sampled to maintain reasonable performance

---

## Pattern Fitting Mechanism (Deep Dive)

The `cutting_edge/pattern_fitting_module.py` is the core intelligence of the application, responsible for solving the **2D irregular packing problem**. It figures out how to optimally arrange garment patterns onto a piece of cloth (which may be a scrap/remnant with holes) to maximize material utilization.

### High-Level Architecture

The module is built around one main class, **`PatternFittingModule`**, which handles the geometry, logic, and optimization strategies. It optionally uses a **`PlacementOptimizer`** neural network to "guess" good placements.

*   **Inputs**: A list of `Pattern` objects (shapes to cut) and a `ClothMaterial` object (fabric canvas).
*   **Outputs**: Valid (x, y) coordinates, rotation, and flip status for each pattern.
*   **Key Libraries**:
    *   **Shapely**: Handles complex geometry (polygon interactions, overlaps, containment).
    *   **PyTorch**: Runs the neural network for placement suggestions.
    *   **Matplotlib**: Visualizes the final layout.

### The Fitting Workflow (`fit_patterns`)

The main entry point is the `fit_patterns` method. It follows this sequence:

1.  **Auto-Scaling (Optional)**: If enabled (via `AUTO_SCALE` in config), runs a binary search simulation to find the largest scale factor that allows patterns to fit.
2.  **Pre-Filtering**: Discards patterns mathematically impossible to fit (larger than cloth even when rotated).
3.  **Sorting**: Patterns sorted by **Area (Descending)** - big pieces first is standard bin-packing heuristic.
4.  **Sequential Placement**: Iterates through sorted patterns, calling `find_best_placement` for each.

### Placement Logic (`find_best_placement`)

Employs a **hybrid search strategy**:

- **Neural Suggestion**: If enabled, `PlacementOptimizer` predicts a "best guess" position.
- **Heuristic Search (BLF)**: Uses Bottom-Left-Fill algorithm (Jakobs, 1996) for remnant cloths.
- **Grid Search**: Tests grid of positions with multiple rotations. **Early Stopping**: Stops if score exceeds `EXCELLENT_SCORE_THRESHOLD` (15.0).
- **Dense Fallback**: If heuristic fails, triggers `_find_baseline_placement` with 5.0cm resolution.

### Validation & Scoring

- **Validity Check**: Containment (100% inside cloth), no overlaps, avoids defects.
- **Scoring**: Edge bonus, utilization bonus, compactness bonus, grain alignment, gap penalty.

### Key Features

*   **Remnant Aware**: Handles non-rectangular cloth scraps.
*   **Defect Avoidance**: Navigates around holes/stains.
*   **Multi-Strategy**: Combines Neural AI, Heuristic BLF, and Grid search.
*   **Dynamic**: Resizes patterns on the fly to maximize material usage.