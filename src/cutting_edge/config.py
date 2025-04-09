"""
Configuration constants for the Cutting Edge project.

This module contains all the configuration constants used across different modules
in the project. These constants are organized by their respective domains.
"""

# Image Processing Constants
IMAGE_PROCESSING = {
    "STANDARD_IMAGE_SIZE": 224,  # Standard size for model input
    "MAX_CLOTH_SIZE": 1024,  # Maximum size for cloth images
    "MIN_DIMENSION": 32,  # Minimum dimension for any image
    "IMAGENET_MEAN": [0.485, 0.456, 0.406],  # ImageNet mean values
    "IMAGENET_STD": [0.229, 0.224, 0.225],  # ImageNet std values
    "EDGE_DETECTION_LOW": 50,  # Lower threshold for Canny edge detection
    "EDGE_DETECTION_HIGH": 150,  # Higher threshold for Canny edge detection
    "CONTOUR_MIN_AREA": 100,  # Minimum area for valid contours
    "MORPH_KERNEL_SIZE": (5, 5),  # Kernel size for morphological operations
    "MORPH_CLOSE_ITERATIONS": 2,  # Number of iterations for closing operation
    "DEFAULT_PREPROCESS_SIZE": (512, 512),  # Default size for preprocessing images
    "MIN_AREA_RATIO": 0.01,  # Minimum area ratio for contour filtering
    "HSV_SATURATION_THRESHOLD": 0.05,  # Threshold for HSV saturation detection
    "ADAPTIVE_THRESHOLD_BLOCK_SIZE": 11,  # Block size for adaptive thresholding
    "ADAPTIVE_THRESHOLD_C": 2,  # Constant subtracted from mean for adaptive thresholding
    "MARGIN_RATIO": 0.1,  # Ratio for margin in fallback rectangle
    "CONTOUR_APPROX_EPSILON": 0.02,  # Epsilon for contour approximation
    "CONTOUR_MIN_PERIMETER": 100,  # Minimum perimeter for valid contours
    "CONTOUR_MAX_APPROX_POINTS": 100,  # Maximum number of points in approximated contour
    "DEFAULT_PATTERN_WIDTH": 50,  # Default pattern width for manual placement
    "DEFAULT_PATTERN_HEIGHT": 50,  # Default pattern height for manual placement
}

# Model Architecture Constants
MODEL = {
    "DEFAULT_ENCODER": "resnet50",  # Default encoder for models
    "DEFAULT_CLOTH_TYPES": 10,  # Default number of cloth types
    "FEATURE_DIM": 512,  # Dimension of feature vectors
    "HIDDEN_DIM": 256,  # Hidden layer dimension
    "DROPOUT_RATE": 0.5,  # Dropout rate for regularization
    "DEFAULT_FLATTENED_SIZE": 2048,  # Default size after flattening
    "LSTM_HIDDEN_SIZE": 256,  # Hidden size for LSTM layers
    "LSTM_NUM_LAYERS": 2,  # Number of LSTM layers
    "LSTM_BIDIRECTIONAL": True,  # Whether to use bidirectional LSTM
    # Pattern Recognition specific constants
    "DIMENSION_PREDICTOR_DROPOUT_1": 0.3,  # First dropout rate for dimension predictor
    "DIMENSION_PREDICTOR_DROPOUT_2": 0.2,  # Second dropout rate for dimension predictor
    "MIN_CORNER_POINTS": 4,  # Minimum number of corner points needed
}

# Training Constants
TRAINING = {
    "BATCH_SIZE": 32,  # Batch size for training
    "LEARNING_RATE": 0.001,  # Learning rate for optimizers
    "DEFAULT_EPOCHS": 50,  # Default number of training epochs
    "NUM_WORKERS": 4,  # Number of workers for data loading
    "VALIDATION_SPLIT": 0.2,  # Validation split ratio
    "EARLY_STOPPING_PATIENCE": 5,  # Patience for early stopping
    "WEIGHT_DECAY": 0.0001,  # Weight decay for regularization
    "GRADIENT_CLIP_NORM": 1.0,  # Gradient clipping norm
    # Pattern Recognition specific training constants
    "EARLY_STOPPING_MIN_DELTA": 0.001,  # Minimum improvement for early stopping
    "LR_SCHEDULER_FACTOR": 0.5,  # Factor to reduce learning rate by
    "LR_SCHEDULER_PATIENCE": 3,  # Patience for learning rate scheduler
    "CHECKPOINT_DIR": "checkpoints",  # Directory for saving checkpoints
    "CHECKPOINT_FREQUENCY": 1,  # Save checkpoint every N epochs
}

# Data Augmentation Constants
AUGMENTATION = {
    "RANDOM_HORIZONTAL_FLIP_PROB": 0.5,  # Probability of horizontal flip
    "RANDOM_ROTATION_DEGREES": 10,  # Maximum rotation degrees
    "COLOR_JITTER_BRIGHTNESS": 0.2,  # Brightness jitter factor
    "COLOR_JITTER_CONTRAST": 0.2,  # Contrast jitter factor
    "RANDOM_AFFINE_TRANSLATE": 0.1,  # Translation factor for random affine
}

# Pattern Fitting Constants
PATTERN_FITTING = {
    "ROTATION_ANGLES": [0, 90, 180, 270],  # Allowed rotation angles
    "GRID_SIZE": 32,  # Size of the grid for pattern placement
    "MARGIN": 10,  # Margin around patterns
    "MAX_FAILURES": 100,  # Maximum number of placement failures
    "INVALID_PLACEMENT_PENALTY": -1.0,  # Penalty for invalid placements
    "SMALL_PENALTY": -0.5,  # Penalty for small placements
    "SMALL_REWARD": 0.1,  # Reward for successful small placements
    "MAX_POSITIONS_TO_TRY": 100,  # Maximum positions to try for placement
    "MIN_OVERLAP_RATIO": 0.1,  # Minimum overlap ratio for valid placement
    "REWARD_SCALING": 1.0,  # Scaling factor for rewards
    "PATTERN_SCALE_FACTOR": 0.2,  # Scale factor for pattern dimensions
    "PATTERN_MARGIN_X": 0.1,  # X-axis margin for pattern placement
    "PATTERN_MARGIN_Y": 0.1,  # Y-axis margin for pattern placement
    "MAX_STEPS": 50,  # Maximum steps per episode
    "MAX_INFERENCE_STEPS": 1000,  # Maximum steps during inference
    "MAX_FAILURES_BEFORE_MANUAL": 10,  # Maximum failures before manual placement
}

# Visualization Constants
VISUALIZATION = {
    "FIGURE_SIZE": (15, 10),  # Default figure size
    "DPI": 300,  # DPI for saved figures
    "COLOR_MAP": "viridis",  # Default color map
    "CONTOUR_COLOR": (0, 255, 0),  # Color for contours (BGR)
    "EDGE_COLOR": (0, 0, 255),  # Color for edges (BGR)
    "TEXT_COLOR": (255, 255, 255),  # Color for text (BGR)
    "BACKGROUND_COLOR": (240, 240, 240),  # Background color (BGR)
    "PATTERN_COLORS": [  # Colors for different patterns
        "red", "blue", "green", "purple", "orange",
        "cyan", "magenta", "yellow", "pink", "brown"
    ],
}

# Environment Constants
ENVIRONMENT = {
    "DEVICE": "cuda",  # Default device for PyTorch
    "RANDOM_SEED": 42,  # Random seed for reproducibility
    "LOG_LEVEL": "INFO",  # Default logging level
    "TENSORBOARD_LOG_DIR": "./logs/",  # Directory for tensorboard logs
    "MODEL_SAVE_DIR": "models/",  # Directory for saving models
    "DATA_DIR": "data/",  # Directory for data
    "OUTPUT_DIR": "outputs/",  # Directory for outputs
}

# Dataset Constants
DATASET = {
    "TRAIN_RATIO": 0.7,  # Training data ratio
    "VAL_RATIO": 0.15,  # Validation data ratio
    "TEST_RATIO": 0.15,  # Test data ratio
    "RANDOM_SEED": 42,  # Random seed for reproducibility
    "AUGMENTATION_PROB": 0.5,  # Probability of augmentation
    "MAX_PATTERNS_PER_CLOTH": 10,  # Maximum patterns per cloth
    "MIN_PATTERN_SIZE": 32,  # Minimum pattern size
    "MAX_PATTERN_SIZE": 512,  # Maximum pattern size
    "DEFAULT_PATTERN_DIMENSIONS": [256, 256],  # Default pattern dimensions
    "DEFAULT_IMAGE_SIZE": (512, 512),  # Default image size for preprocessing
    "DEFAULT_EMPTY_IMAGE": (3, 512, 512),  # Default empty image dimensions
    "DEFAULT_EMPTY_DIMENSIONS": [256.0, 256.0],  # Default empty dimensions
} 