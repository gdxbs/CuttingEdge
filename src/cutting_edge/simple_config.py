"""
Simple Configuration File
Centralized configuration for the simplified cutting edge system.
All magic numbers are here for easy modification.
"""

# Pattern Recognition Configuration
PATTERN = {
    # Default dimensions when not found in filename
    "DEFAULT_WIDTH": 100,  # cm
    "DEFAULT_HEIGHT": 150,  # cm
    # Image processing
    "IMAGE_SIZE": 256,  # Size for neural network input
    "PIXEL_TO_CM": 0.1,  # Conversion factor for OpenCV estimation
    # Pixel to cm conversion
    "PIXEL_TO_CM": 0.1,  # Conversion factor for OpenCV estimation
    # Pattern types (for future multi-class support)
    "TYPES": ["shirt", "pants", "dress", "sleeve", "collar", "other"],
    # Data augmentation (for training)
    "AUGMENTATION": {
        "enabled": True,
        "rotation_degrees": 10,
        "horizontal_flip_prob": 0.5,
        "brightness": 0.2,
        "contrast": 0.2,
    },
    # Contour detection
    "MIN_CONTOUR_AREA": 100,  # Minimum area for valid contours
    "CONTOUR_APPROX_EPSILON": 0.02,  # Epsilon for contour approximation
}

# Cloth Recognition Configuration
CLOTH = {
    # Default dimensions
    "DEFAULT_WIDTH": 200,  # cm
    "DEFAULT_HEIGHT": 300,  # cm
    # Safety margins
    "EDGE_MARGIN": 5,  # cm from edges
    # Image processing
    "IMAGE_SIZE": 256,
    # Cloth detection thresholds
    "HSV_LOWER": [0, 0, 50],  # Lower HSV bound for cloth detection
    "HSV_UPPER": [180, 255, 255],  # Upper HSV bound
    # Pixel to cm conversion
    "PIXEL_TO_CM": 0.1,
    # Morphological operations
    "MORPH_KERNEL_SIZE": 5,
    "MORPH_ITERATIONS": 2,
    # Edge detection (Canny)
    "CANNY_LOW": 50,
    "CANNY_HIGH": 150,
    # Cloth types (for future classification)
    "TYPES": ["cotton", "silk", "wool", "polyester", "mixed", "other"],
}

# Pattern Fitting Configuration
FITTING = {
    # Grid settings
    "GRID_SIZE": 10,  # Divide cloth into NxN grid
    # Placement attempts
    "MAX_ATTEMPTS": 100,  # Max attempts to place each pattern
    # Rotation angles to try
    "ROTATION_ANGLES": [0, 90, 180, 270],  # degrees
    # Reward system
    "REWARDS": {
        "overlap_penalty": -10,
        "edge_bonus": 2,  # Bonus for placing near edges
        "utilization_bonus": 5,  # Bonus per % of cloth used
        "compactness_bonus": 3,  # Bonus for compact placement
        "gap_penalty": -1,  # Penalty for creating small gaps
    },
    # Minimum gap size (to avoid unusable spaces)
    "MIN_GAP_SIZE": 20,  # cm
    # Minimum pattern coverage
    "MIN_PATTERN_COVERAGE": 0.95,  # 95% of pattern must be within cloth
    # Grid sample size for placement search
    "GRID_SAMPLE_SIZE": 100,
    # RL Agent settings
    "AGENT": {
        "learning_rate": 0.001,
        "hidden_size": 64,
        "state_size": 10,  # Size of state representation
    },
}

# Training Configuration
TRAINING = {
    # Data split
    "TRAIN_RATIO": 0.8,
    # Training parameters
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "LEARNING_RATE": 0.001,
    # Learning rate scheduling
    "LR_SCHEDULER": {
        "enabled": True,
        "factor": 0.5,  # Reduce LR by this factor
        "patience": 3,  # Epochs to wait before reduction
    },
    # Early stopping
    "EARLY_STOPPING": {
        "enabled": True,
        "patience": 5,
        "min_delta": 0.001,
    },
    # Checkpointing
    "CHECKPOINT": {
        "enabled": True,
        "frequency": 5,  # Save every N epochs
        "best_only": True,  # Save only best model
    },
}

# Visualization Configuration
VISUALIZATION = {
    # Colors for pattern visualization (RGB)
    "PATTERN_COLORS": [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [128, 0, 128],  # Purple
        [255, 165, 0],  # Orange
    ],
    # Plot settings
    "FIGURE_SIZE": (10, 10),
    "DPI": 150,
    "FONT_SIZE": 8,
    "LINE_WIDTH": 2,
    "ALPHA": 0.7,  # Transparency for patterns
}

# System Configuration
SYSTEM = {
    # Paths
    "BASE_DIR": "/Users/aryaminus/Developer/cutting-edge",
    "MODELS_DIR": "models",
    "OUTPUT_DIR": "output",
    "DATA_DIR": "data",
    "IMAGES_DIR": "images",
    # Logging
    "LOG_LEVEL": "INFO",
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # Random seed for reproducibility
    "RANDOM_SEED": 42,
    # Device selection
    "USE_GPU": True,  # Use GPU if available
}

# ImageNet normalization (for better model performance)
IMAGENET_NORMALIZE = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
