"""
Simple Pattern Recognition Module
This module handles pattern (shape) image processing and dimension extraction.
It's designed to be beginner-friendly with clear comments explaining each step.
"""

import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Simple data class to hold pattern information"""

    id: int
    width: float
    height: float
    contour: np.ndarray
    name: str


class SimplePatternCNN(nn.Module):
    """
    A simple CNN for pattern feature extraction.
    Uses basic conv layers instead of complex pre-trained models.
    """

    def __init__(self):
        super().__init__()
        # Simple CNN architecture that's easy to understand
        self.features = nn.Sequential(
            # First conv block: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second conv block: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Third conv block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Adaptive pooling to get fixed size output
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Dimension predictor - predicts width and height
        self.dimension_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),  # Output: [width, height]
        )

    def forward(self, x):
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten

        # Predict dimensions
        dimensions = self.dimension_head(features)

        return {"features": features, "dimensions": dimensions}


class PatternProcessor:
    """
    Handles all pattern processing tasks including:
    - Loading images from folders
    - Extracting dimensions from filenames
    - Processing images for the model
    - Managing model training and inference
    """

    # Magic numbers (default values) - easy to modify
    DEFAULT_WIDTH = 100  # Default pattern width in cm
    DEFAULT_HEIGHT = 150  # Default pattern height in cm
    IMAGE_SIZE = 256  # Size to resize images for processing

    def __init__(self, model_path: str = "models/pattern_model.pth"):
        self.model_path = model_path
        self.model = SimplePatternCNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"Pattern Processor initialized. Using device: {self.device}")

    def extract_dimensions_from_filename(self, filename: str) -> Tuple[float, float]:
        """
        Extract dimensions from filename format: pattern_widthxheight.jpg
        Falls back to defaults if not found.
        """
        # Try to find pattern like "100x150" in filename
        match = re.search(r"(\d+)x(\d+)", filename)
        if match:
            width = float(match.group(1))
            height = float(match.group(2))
            logger.info(f"Extracted dimensions from {filename}: {width}x{height}")
            return width, height
        else:
            logger.warning(
                f"No dimensions found in {filename}, using defaults: {self.DEFAULT_WIDTH}x{self.DEFAULT_HEIGHT}"
            )
            return self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT

    def estimate_dimensions_opencv(self, image_path: str) -> Tuple[float, float]:
        """
        Estimate pattern dimensions using OpenCV (contour detection).
        This is a fallback when dimensions aren't in filename.
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Get the largest contour (assumed to be the pattern)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Convert pixel dimensions to cm (assuming some scale)
            # This is a simple heuristic - adjust based on your needs
            pixel_to_cm = 0.1  # 1 pixel = 0.1 cm (adjustable)
            width = w * pixel_to_cm
            height = h * pixel_to_cm

            logger.info(f"OpenCV estimated dimensions: {width:.1f}x{height:.1f} cm")
            return width, height

        return self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT

    def process_image(self, image_path: str) -> Dict:
        """
        Process a single pattern image and extract all information.
        """
        logger.info(f"Processing pattern image: {image_path}")

        # Extract dimensions from filename first
        filename = os.path.basename(image_path)
        width, height = self.extract_dimensions_from_filename(filename)

        # If defaults were used, try OpenCV estimation
        if width == self.DEFAULT_WIDTH and height == self.DEFAULT_HEIGHT:
            width, height = self.estimate_dimensions_opencv(image_path)

        # Load and preprocess image for neural network
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)

        # Extract contour for later use in fitting
        contour = self.extract_contour(image_path)

        return {
            "filename": filename,
            "path": image_path,
            "dimensions": (width, height),
            "model_features": output["features"].cpu().numpy(),
            "model_dimensions": output["dimensions"].cpu().numpy(),
            "contour": contour,
            "original_image": img,
        }

    def extract_contour(self, image_path: str) -> np.ndarray:
        """
        Extract the contour of the pattern for fitting purposes.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Return the largest contour
            return max(contours, key=cv2.contourArea)
        return np.array([])

    def load_model(self) -> bool:
        """
        Load pre-trained model if it exists.
        """
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing pattern model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Pattern model loaded successfully!")
            return True
        else:
            logger.warning(f"No existing model found at {self.model_path}")
            return False

    def save_model(self):
        """
        Save the current model state.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
            },
            self.model_path,
        )
        logger.info(f"Pattern model saved to {self.model_path}")
