# Pattern Fitting Module Explanation

The `cutting_edge/pattern_fitting_module.py` is the core intelligence of the application, responsible for solving the **2D irregular packing problem**. It figures out how to optimally arrange garment patterns onto a piece of cloth (which may be a scrap/remnant with holes) to maximize material utilization.

## 1. High-Level Architecture

The module is built around one main class, **`PatternFittingModule`**, which handles the geometry, logic, and optimization strategies. It optionally uses a **`PlacementOptimizer`** neural network to "guess" good placements.

*   **Inputs**: A list of `Pattern` objects (shapes to cut) and a `ClothMaterial` object (fabric canvas).
*   **Outputs**: Valid (x, y) coordinates, rotation, and flip status for each pattern.
*   **Key Libraries**:
    *   **Shapely**: Handles complex geometry (polygon interactions, overlaps, containment).
    *   **PyTorch**: Runs the neural network for placement suggestions.
    *   **Matplotlib**: Visualizes the final layout.

---

## 2. The Fitting Workflow (`fit_patterns`)

The main entry point is the `fit_patterns` method. It follows this sequence:

1.  **Auto-Scaling (Optional)**:
    *   If enabled, it calls `find_optimal_scale` to run a binary search simulation.
    *   It determines if patterns can be scaled up (e.g., 20% larger) while still fitting on the cloth, maximizing usage.

2.  **Pre-Filtering**:
    *   Checks every pattern against the cloth dimensions.
    *   If a pattern is mathematically impossible to fit (larger than the cloth even when rotated), it is discarded immediately to save processing time.

3.  **Sorting**:
    *   Patterns are typically sorted by **Area (Descending)**.
    *   Placing big pieces first is a standard bin-packing heuristic because small pieces can easily fit in the gaps left by big ones.

4.  **Sequential Placement**:
    *   Iterates through the sorted patterns one by one and calls `find_best_placement` for each.
    *   Once a pattern is placed, it is added to `existing_placements` and becomes an obstacle for subsequent patterns.

---

## 3. Placement Logic (`find_best_placement`)

This function determines exactly where a single pattern goes. It employs a **hybrid search strategy**:

### A. Neural Suggestion (The "Smart" Way)
*   If enabled, the `PlacementOptimizer` network analyzes the current state (cloth fullness, pattern shape).
*   It predicts a "best guess" position, rotation, and flip.
*   This is added to the priority list of positions to try.

### B. Heuristic Search (Bottom-Left-Fill)
*   If the cloth is detected as a "remnant" (irregular shape), it adds positions based on the **Bottom-Left-Fill (BLF)** algorithm (Jakobs, 1996).
*   It specifically tries placing the pattern:
    *   In the bottom-left corner.
    *   Immediately adjacent to the right or top of just-placed pieces.

### C. Grid Search (The "Robust" Way)
*   It generates a grid of potential coordinates (controlled by `GRID_SIZE`).
*   At every grid point, it tests multiple rotations (defined in config, e.g., 0°, 90°, 180°).
*   **Optimization (Early Stopping)**: If it finds a placement with a "Score" strictly higher than `EXCELLENT_SCORE_THRESHOLD` (e.g., >15.0), it stops searching immediately and takes that spot.

### D. Dense Fallback (The "Safety Net")
*   If the heuristic/grid search fails efficiently (due to sparse checking), it enters a `_find_baseline_placement` mode.
*   This performs a **dense search** (e.g., every 5.0cm) with strict rotations (0, 90, 180, 270) to ensure that if a spot exists, it will be found.

---

## 4. Validation & Scoring

For every potential position, the module runs two critical checks:

### 1. Validity Check (`is_valid_placement`)
*   **Containment**: Is the pattern 100% inside the cloth polygon?
*   **Overlap**: Does it intersect with any previously placed patterns? (Allows tiny overlap tolerance).
*   **Defects**: Does it touch any holes, stains, or tears defined in the cloth?

### 2. Scoring (`calculate_placement_score`)
If placement is valid, it gets a score based on:
*   **Utilization**: Bonus for using more area.
*   **Compactness**: Bonus for being close to other pieces (clustering).
*   **Gap Penalty**: Penalty for creating unusable void spaces (holes in the layout).

---

## 5. Visualization & Reporting

After fitting is done, the `visualize` method generates the output:
*   **Visual**: Creates a `.png` showing the cloth (blue), defects (red hatched zones), and placed patterns (rainbow colors).
*   **Report**: Writes a human-readable text file summarizing utilization percentage, waste area, and details of every placed piece.

---

## Summary of Key Features
*   **Remnant Aware**: Specifically designed to handle non-rectangular cloth scraps.
*   **Defect Avoidance**: Smartly navigates around holes/stains.
*   **Multi-Strategy**: Combines Neural AI, Heuristic BLF, and Brute-force Grid search for robustness.
*   **Dynamic**: Can resize patterns on the fly to maximize material usage.
