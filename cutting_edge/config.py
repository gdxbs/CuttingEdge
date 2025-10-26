"""
Centralized configuration for the Cutting Edge pattern fitting system.
All magic numbers and settings are defined here for easy modification.
"""

import os

# System settings
SYSTEM = {
    "BASE_DIR": os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "IMAGES_DIR": "images",
    "PATTERN_DIR_NAME": "shape",
    "CLOTH_DIR_NAME": "shape",
    "IMAGE_EXTENSIONS": ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"],
    "MODELS_DIR": "models",
    "OUTPUT_DIR": "output",
    "DATA_DIR": "data",
    "LOG_LEVEL": "INFO",
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "RANDOM_SEED": 42,
    "USE_GPU": True,  # Will fall back to CPU if GPU not available
}

# ImageNet normalization constants for transfer learning
IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# Pattern recognition settings
PATTERN = {
    # Default dimensions (cm) if not specified in filename
    "DEFAULT_WIDTH": 60,
    "DEFAULT_HEIGHT": 80,
    # Standard pattern dimensions by type (cm)
    "STANDARD_DIMENSIONS": {
        "shirt": {"width": 60, "height": 80},
        "pants": {"width": 45, "height": 110},
        "dress": {"width": 55, "height": 120},
        "sleeve": {"width": 25, "height": 60},
        "collar": {"width": 40, "height": 15},
        "other": {"width": 50, "height": 50},
    },
    # Pattern types for classification
    "TYPES": ["shirt", "pants", "dress", "sleeve", "collar", "other"],
    # Image processing
    "IMAGE_SIZE": 256,  # Size to resize images to
    "PIXEL_TO_CM": 0.105,  # Conversion factor for manual detection
    "CONTOUR_SIMPLIFICATION": 0.01,  # Contour simplification factor (% of perimeter)
    # CNN Architecture (feature extraction)
    "BACKBONE": "resnet18",  # Options: "simple", "resnet18", "efficientnet-b0"
    "FEATURE_DIM": 512,  # Feature dimension for resnet18
    "ESTIMATOR_HIDDEN_DIM": 128,
    # CV settings
    "ADAPTIVE_THRESH_BLOCK_SIZE": 11,
    "ADAPTIVE_THRESH_C": 2,
    "MORPH_KERNEL_SIZE": (5, 5),
    "MAX_CORNERS": 8,
    "CORNER_QUALITY_LEVEL": 0.01,
    "CORNER_MIN_DISTANCE": 10,
    # Data augmentation (for training)
    "AUGMENTATION": {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation": 10,  # degrees
        "brightness": 0.1,  # factor
        "contrast": 0.1,  # factor
        "scale": 0.1,  # factor
    },
}

# Cloth recognition settings
CLOTH = {
    # Default dimensions (cm) if not specified in filename
    "DEFAULT_WIDTH": 200,
    "DEFAULT_HEIGHT": 300,
    # Image processing
    "IMAGE_SIZE": 512,
    "PIXEL_TO_CM": 0.1,
    "EDGE_MARGIN": 5,  # Safety margin from edges (cm)
    # Segmentation
    "USE_UNET": True,  # Whether to use U-Net or color-based segmentation
    # Color-based segmentation (HSV)
    "HSV_LOWER": [0, 0, 0],  # Lower bound for cloth detection
    "HSV_UPPER": [180, 255, 240],  # Upper bound for cloth detection
    "MORPH_KERNEL_SIZE": 5,
    "MORPH_ITERATIONS": 2,
    # U-Net settings
    "UNET_IN_CHANNELS": 3,
    "UNET_OUT_CHANNELS": 2,
    # CV settings
    "THRESHOLD_VALUE": 127,
    "MIN_DEFECT_AREA": 10,
    # Material types
    "TYPES": ["cotton", "silk", "wool", "polyester", "mixed", "other"],
    # Heuristics for cloth type detection
    "TYPE_HEURISTICS": {
        "saturation_threshold": 30,
        "value_threshold": 200,
        "hue_red_lower": 30,
        "hue_red_upper": 150,
        "saturation_vibrant": 100,
    },
    # Gabor filter settings for texture analysis
    "GABOR_SETTINGS": {
        "ksize": (21, 21),
        "sigma": 5,
        "lambd": 10,
        "gamma": 0.5,
        "psi": 0,
        "orientations": 4,
    },
}

# Pattern fitting settings
FITTING = {
    # Placement optimization
    "GRID_SIZE": 5,  # Grid divisions for placement search
    "MAX_ATTEMPTS": 200,  # Max attempts to place a pattern
    "ROTATION_ANGLES": [0, 90, 180, 270],  # Angles to try (degrees)
    "ALLOW_FLIPPING": True,  # Whether to try flipped patterns
    "MIN_PATTERN_COVERAGE": 0.95,  # Minimum pattern coverage by cloth (%)
    "OVERLAP_TOLERANCE": 0.01,  # Allowable overlap between patterns (%)
    "GRID_SAMPLE_SIZE": 200,  # Max grid points to sample
    "MIN_GAP_SIZE": 20,  # Minimum useful gap size (cm)
    # Rewards system
    "REWARDS": {
        "overlap_penalty": -20,  # Penalty for overlapping patterns
        "edge_bonus": 5,  # Bonus for placing near edges
        "utilization_bonus": 10,  # Bonus for good material utilization
        "compactness_bonus": 7,  # Bonus for placing patterns together
        "gap_penalty": -2,  # Penalty for creating small gaps
        "origin_bonus": 3,  # Bonus for placing patterns near the origin
    },
    # Optimization level
    "USE_NEURAL_OPTIMIZER": True,  # Whether to use neural network or just grid search
    # RL Agent settings (if using neural optimizer)
    "AGENT": {
        "learning_rate": 0.001,
        "hidden_dim": 64,
        "state_dim": 20,
        "action_dim": 4,  # x, y, rotation_idx, flip
    },
}

# Visualization settings
VISUALIZATION = {
    "FIGURE_SIZE": (12, 10),
    "DPI": 150,
    "LINE_WIDTH": 2,
    "FONT_SIZE": 10,
    "ALPHA": 0.7,
}

# Training settings
TRAINING = {
    "BATCH_SIZE": 16,
    "EPOCHS": 50,
    "LEARNING_RATE": 0.001,
    "TRAIN_RATIO": 0.8,  # Train/test split ratio
    "VALIDATION_STEPS": 10,  # Validate every N steps
    "EARLY_STOPPING": 10,  # Early stopping patience (epochs)
    "SAVE_BEST_ONLY": True,  # Only save best model
}
