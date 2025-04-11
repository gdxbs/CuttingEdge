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
    "MAX_FAILURES": 100,  # Maximum number of placement failures
    "MAX_POSITIONS_TO_TRY": 100,  # Maximum positions to try for placement
    "PATTERN_SCALE_FACTOR": 0.5,  # Scale factor for pattern dimensions
    "PATTERN_MARGIN_X": 0.1,  # X-axis margin for pattern placement
    "PATTERN_MARGIN_Y": 0.1,  # Y-axis margin for pattern placement
    "MAX_STEPS": 50,  # Maximum steps per episode
    "MAX_INFERENCE_STEPS": 1000,  # Maximum steps during inference
    "MAX_FAILURES_BEFORE_MANUAL": 10,  # Maximum failures before manual placement
    "CLOTH_BOUNDARY_MARGIN": 0.9,  # Percentage of cloth to use for boundary margins
    "MIN_PATTERN_COVERAGE": 0.95,  # Minimum percentage of pattern that must be within cloth
    "DEFAULT_CLOTH_WIDTH": 512,  # Default cloth width
    "DEFAULT_CLOTH_HEIGHT": 512,  # Default cloth height
    "MIN_PATTERN_WIDTH_RATIO": 0.05,  # Minimum pattern width as ratio of cloth width
    "MIN_PATTERN_HEIGHT_RATIO": 0.05,  # Minimum pattern height as ratio of cloth height
    "GRID_SAMPLE_SIZE": 100,  # Number of sample points for grid placement
    "SPIRAL_ANGLE_STEP": 15,  # Angle step for spiral pattern placement
    "SPIRAL_RADIUS_STEP": 0.1,  # Radius step for spiral pattern placement as ratio of max radius
}

# Visualization Constants
VISUALIZATION = {
    "FIGURE_SIZE": (15, 10),  # Default figure size
    "DPI": 300,  # DPI for saved figures
    "COLOR_MAP": "viridis",  # Default color map
    "CONTOUR_COLOR": (0, 255, 0),  # Color for contours (BGR)
    "EDGE_COLOR": (0, 0, 255),  # Color for edges (BGR)
    "PATTERN_COLORS": [  # Colors for different patterns
        "red", "yellow", "green", "purple", "orange",
        "cyan", "magenta", "blue", "pink", "brown"
    ],
}

# Environment Constants
ENVIRONMENT = {
    "DEVICE": "cuda",  # Default device for PyTorch
    "LOG_LEVEL": "INFO",  # Default logging level
    "TENSORBOARD_LOG_DIR": "./logs/",  # Directory for tensorboard logs
}

# Dataset Constants
DATASET = {
    "DEFAULT_PATTERN_DIMENSIONS": [256, 256],  # Default pattern dimensions
    "DEFAULT_IMAGE_SIZE": (512, 512),  # Default image size for preprocessing
    "DEFAULT_EMPTY_IMAGE": (3, 512, 512),  # Default empty image dimensions
    "DEFAULT_EMPTY_DIMENSIONS": [256.0, 256.0],  # Default empty dimensions
} 