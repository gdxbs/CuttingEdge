# Technical Synopsis: Cutting Edge Pattern Fitting System

## 1. Overview
The **Cutting Edge** system is a sophisticated computer vision and optimization platform designed to minimize fabric waste in the garment industry. It automates the process of fitting garment patterns (e.g., shirts, sleeves) onto cloth materials, including irregular remnants. The system balances robustness with simplicity, leveraging a hybrid approach of classical computer vision, heuristics, and optional neural networks.

## 2. Core Workflow
The end-to-end workflow operates in three sequential stages:

1.  **Pattern Recognition**:
    -   **Input**: Pattern images (raster or SVG).
    -   **Processing**: Extracts contours, detects key points, and classifies the pattern type (e.g., "shirt") using a CNN backbone.
    -   **Output**: A standardized `Pattern` object with physical dimensions and shape data.

2.  **Cloth Recognition**:
    -   **Input**: Cloth material images.
    -   **Processing**: Segments the usable cloth area from the background, identifies defects (holes, stains), and analyzes material properties (e.g., grain direction).
    -   **Output**: A `ClothMaterial` object defining the usable polygon and exclusion zones (defects).

3.  **Pattern Fitting (Optimization)**:
    -   **Input**: List of `Pattern` objects and one `ClothMaterial` object.
    -   **Processing**: Places patterns onto the cloth using geometric algorithms (Bottom-Left-Fill) and heuristic search, optimizing for material utilization and adherence to constraints (e.g., no overlaps, defect avoidance).
    -   **Output**: A placement layout, utilization metrics, and visual reports.

## 3. Methodologies & Algorithms

### Computer Vision
-   **Contour Extraction**: Uses OpenCV for adaptive thresholding and morphological operations to isolate shapes.
-   **Defect Detection**: A multi-faceted approach detecting:
    -   **Holes**: Dark region thresholding.
    -   **Stains**: Statistical intensity/color variance analysis in HSV space.
    -   **Tears/Edges**: Gradient magnitude analysis using Sobel filters.
-   **Texture Analysis**: Gabor filters are employed to estimate fabric grain direction.
-   **Neural Networks (Optional)**:
    -   **Pattern Classification**: A configurable CNN (ResNet18, EfficientNet-B0) classifies patterns and estimates dimensions.
    -   **Segmentation**: A U-Net architecture can be used for pixel-perfect cloth/defect segmentation.

### Optimization & Geometry
-   **Shapely**: The core geometric engine. All patterns and cloth areas are converted to `shapely.geometry.Polygon` objects for precise boolean operations (intersection, difference, containment).
-   **Heuristic Search**:
    -   **Grid Search**: Systematically tests positions across the cloth surface.
    -   **Bottom-Left-Fill (BLF)**: A classic packing heuristic that prioritizes placing items as close to the bottom-left as possible.
    -   **Scoring Function**: Placements are evaluated based on a weighted sum of:
        -   Material utilization.
        -   Compactness (proximity to other patterns).
        -   Edge alignment (bonus for placing on edges).
        -   Grain alignment.
        -   Penalties for gaps and overlaps.
-   **Reinforcement Learning (Experimental)**: A neural `PlacementOptimizer` can suggest initial placements to guide the heuristic search, potentially speeding up convergence.

## 4. Package Dependencies
The system relies on a robust stack of scientific and vision libraries:

-   **`torch` / `torchvision`**: Deep learning framework for the Pattern Recognizer, U-Net Segmenter, and RL Agent.
-   **`opencv-python`**: Primary tool for image processing (thresholding, contours, filtering).
-   **`shapely`**: Essential for all 2D geometric calculations and validity checks.
-   **`numpy`**: High-performance array manipulation.
-   **`matplotlib`**: Generation of visual outputs and charts.
-   **`cairosvg`**: Rasterization of SVG pattern inputs.

## 5. File Analysis

### `src/cutting_edge/main.py`
The orchestration engine. It initializes the three modules, handles data loading (scanning directories), manages train/test splits, and executes the chosen mode (`demo`, `fit`, `train`, `eval`, `all_cloths`). It also handles high-level reporting and metric aggregation.

### `src/cutting_edge/config.py`
The central nervous system of the project. It contains all tunable parameters, widely referenced across modules.
-   **System**: Paths, logging, GPU usage.
-   **Pattern**: Image processing thresholds, CNN backbone selection, standard dimensions.
-   **Cloth**: Segmentation thresholds, defect parameters, U-Net toggle.
-   **Fitting**: Grid size, rotation angles, reward weights for the scoring function.

### `src/cutting_edge/pattern_recognition_module.py`
-   **`Pattern` Dataclass**: The data structure for pattern entities.
-   **`PatternRecognizer`**: A `torch.nn.Module` wrapping the CNN backbone.
-   **`PatternRecognitionModule`**: Handles image loading, preprocessing, inference, and shape extraction logic (contour simplification, scaling).

### `src/cutting_edge/cloth_recognition_module.py`
-   **`ClothMaterial` Dataclass**: Stores cloth geometry and properties.
-   **`UNetSegmenter`**: The PyTorch U-Net implementation.
-   **`ClothRecognitionModule`**: Implements the complex logic for defect detection, color segmentation, and scaling pixel coordinates to real-world centimeters.

### `src/cutting_edge/pattern_fitting_module.py`
-   **`PatternFittingModule`**: The heavy lifter. It contains the `fit_patterns` loop, the `find_best_placement` search logic, and the `calculate_placement_score` function. It manages the conversion between raster logic and vector (Shapely) logic.
-   **`PlacementOptimizer`**: The experimental RL network.

### `src/cutting_edge/training_optimizer.py`
Implements a grid-search based hyperparameter optimizer (`HeuristicOptimizer`). It iterates through combinations of fitting parameters (grid size, rotations) to find the configuration that maximizes material utilization on the training set.

## 6. Configuration & Customization
The system is designed to be highly tweakable via `config.py`. Key customizations include:
-   **Model Backbone**: Switching `PATTERN["BACKBONE"]` to `efficientnet-b0` for speed or `resnet18` for accuracy.
-   **Fitting Precision**: Adjusting `FITTING["GRID_SIZE"]` and `FITTING["ROTATION_ANGLES"]` trades off between processing speed and packing density.
-   **Defect Sensitivity**: Modifying `CLOTH["MIN_DEFECT_AREA"]` allows fine-tuning of what constitutes a "defect."

## 7. Future Improvements
-   **Full NFP Implementation**: While No-Fit Polygons are mentioned in research references, the current implementation relies heavily on grid search. A full NFP implementation would allow for exact geometric placement without a grid.
-   **End-to-End RL**: The current RL agent is a guide. A fully trained RL agent that places patterns sequentially without a heuristic fallback could potentially outperform the grid search.
-   **Real-time Feedback**: Optimizing the pipeline for real-time camera feed analysis could enable augmented reality applications for manual cutting.
