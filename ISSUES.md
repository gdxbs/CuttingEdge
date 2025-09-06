# Cutting Edge Project Issues

This document tracks the progress of the Cutting Edge project.

**Note:** The project has undergone a major refactoring to simplify the architecture, remove heavy dependencies (like `stable-baselines3` and `gymnasium`), and focus on a more robust, geometry-based approach for pattern fitting using `shapely`. Many of the original issues related to the complex Hierarchical Reinforcement Learning (HRL) approach are now obsolete.

## Completed / Obsolete Issues

### Original Plan (Weeks 1-6) - Largely Obsolete

The original sprint plan is no longer relevant due to the architectural changes. Key functionalities were achieved in a more direct way.

-   ✅ **Core Modules Refactored**: All main modules (`pattern_recognition`, `cloth_recognition`, `pattern_fitting`) have been completely rewritten for simplicity, readability, and robustness.
-   ✅ **HRL Module Replaced**: The complex HRL-based placement has been replaced with a more suitable geometric optimization approach, resolving all related implementation issues.
-   ✅ **Dataset Loader Removed**: The `dataset.py` and `utils.py` files were removed, as the new architecture does not require a complex data loading pipeline for its core functionality. This makes the "Dataset Structure Mismatch" issue obsolete.
-   ✅ **Configuration Centralized**: All magic numbers and settings have been moved to `config.py`, improving maintainability.
-   ✅ **Corner/Feature Extraction**: This is now handled by a combination of OpenCV in `pattern_recognition_module` and is sufficient for the current system.
-   ✅ **Basic Training Pipeline**: Placeholder training functions exist in each module, and the main entry point supports a `--mode train` flag.

## Remaining Issues & Future Enhancements

### High Priority

1.  **Implement a Real Training Loop**
    -   The current `train()` functions in the modules are placeholders.
    -   A proper training loop needs to be implemented, especially for the optional `PlacementOptimizer` in the fitting module, likely using a simple supervised or policy gradient approach.
    -   Requires creating a dataset of (state, optimal_action, reward) tuples.

2.  **Create a Comprehensive Test Suite**
    -   Add unit tests for the core functions in each module (e.g., `calculate_placement_score`, `is_valid_placement`).
    -   Add integration tests for the full `run_fitting_task` pipeline.

### Medium Priority

3.  **Enhance Logging & Metric Monitoring**
    -   While logging is implemented, it could be more structured.
    -   Integrate a tool like TensorBoard or Weights & Biases to visualize training progress and fitting results over time.

4.  **Improve Documentation**
    -   Add detailed API documentation (docstrings) for all public methods.
    -   Create a simple tutorial notebook demonstrating the usage of the system.

### Low Priority / Future Features

5.  **Feature: Multi-Cloth Support**
    -   Extend the system to consider multiple available cloth pieces to best fit a given set of patterns.

6.  **Feature: Waste Optimization**
    -   The system currently optimizes placement but doesn't analyze the remaining "waste" pieces. An enhancement could be to identify if leftover pieces are large enough for future use.

7.  **Feature: Material-Specific Adjustments**
    -   The `material_properties` in the `ClothMaterial` object (e.g., grain direction) are extracted but not yet fully used in the fitting score. The scoring function could be enhanced to account for these.
