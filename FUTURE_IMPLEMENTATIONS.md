# Future Implementations and Unused Features

This document outlines features present in the codebase that are currently unused or "dormant," as well as a roadmap for future upgrades to the Cutting Edge system.

## Unused / Dormant Features

These components are implemented in the code but are not currently active in the default execution flow or require additional setup (like training data) to function.

### 1. AI Pattern Recognition
- **Current Status:** Partially Implemented (Stub).
- **Location:** `cutting_edge/pattern_recognition_module.py`
- **Details:** The module contains a `PatternRecognizer` neural network (ResNet/EfficientNet backbone) designed to classify patterns (e.g., "sleeve", "front_panel") and estimate dimensions.
- **Why it's unused:** The system currently relies on "classic" computer vision (contours) and filename parsing (`panel_50x80...`) for accurate dimensions. The `train()` method is a placeholder that saves the model without actual training.
- **Activation:** Requires implementing a full training loop with a labeled dataset of pattern images.

### 2. Neural Placement Optimizer (RL Agent)
- **Current Status:** Implemented but Disabled.
- **Location:** `cutting_edge/pattern_fitting_module.py` -> `PlacementOptimizer` class.
- **Details:** A Reinforcement Learning (RL) agent structure exists to learn optimal pattern placement strategies over time, rather than relying solely on heuristics (like Bottom-Left-Fill).
- **Why it's unused:** The configuration `FITTING["USE_NEURAL_OPTIMIZER"]` defaults to `False`. The logic to train this agent using rewards (utilization, gap minimization) is not fully hooked up to a training loop.
- **Activation:** Set `USE_NEURAL_OPTIMIZER = True` and implement the RL training epoch loop in `main.py`.

### 3. AI Cloth Segmentation (U-Net) Training
- **Current Status:** Implemented but Untrained.
- **Location:** `cutting_edge/cloth_recognition_module.py` -> `UNetSegmenter` class.
- **Details:** A U-Net architecture is implemented for pixel-perfect cloth and defect segmentation.
- **Why it's unused:** While the *inference* code exists, the `train()` method is a stub. The system currently uses color/threshold-based segmentation, which is robust enough for simple backgrounds.
- **Activation:** Requires a dataset of cloth images with pixel-wise ground truth masks for training.

## Roadmap & Potential Upgrades

### Short Term (Optimization & Polish)
- [ ] **Parallel Processing:** The pattern fitting loop processes cloths sequentially. Using Python's `multiprocessing` could significantly speed up "All Cloths" mode.
- [ ] **GPU Acceleration for Heuristics:** The current finding algorithm uses CPU-based geometry checks (Shapely). Moving polygon intersection checks to GPU (e.g., using CUDA-accelerated geometry libraries) would allow for massive grid searches.
- [ ] **Web Interface:** Currently, the system is CLI-only. A simple Streamlit or Flask dashboard could allow users to upload images and see results in real-time.

### Medium Term (AI Activation)
- [ ] **Pattern Classification Training:** Collect a dataset of ~1000 labeled pattern parts and implement the training loop for `PatternRecognitionModule`. This would allow the system to identify "sleeve vs. body" and apply garment-specific rules (e.g., "sleeves must align with grain").
- [ ] **Reinforcement Learning for Packing:** Activate the `PlacementOptimizer`. Train it on thousands of random layouts to learn strategies that beat human heuristics (like nesting small pieces inside holes).

### Long Term (Advanced Features)
- [ ] **3D Integration:** Map 2D patterns to 3D avatars to visualize the final garment before cutting.
- [ ] **Defect-Aware Generative Nesting:** Use Generative AI to "hallucinate" optimal layouts for complex defect patterns (like leather hides) that traditional algorithms struggle with.
- [ ] **Camera Integration:** Real-time camera feed support to detect cloth boundaries and project cutting lines directly onto the fabric table (Augmented Reality).
