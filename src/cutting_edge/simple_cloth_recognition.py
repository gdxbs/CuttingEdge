"""
Simple Cloth Recognition Module
This module handles cloth image processing and dimension extraction.
Designed to be beginner-friendly with clear explanations.
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleClothCNN(nn.Module):
    """
    A simple CNN for cloth analysis.
    Extracts features and predicts cloth properties.
    """

    def __init__(self):
        super().__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Global pooling
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Dimension predictor
        self.dimension_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),  # Output: [width, height]
        )

    def forward(self, x):
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        dimensions = self.dimension_head(features_flat)

        return {"features": features_flat, "dimensions": dimensions}


class ClothProcessor:
    """
    Handles all cloth processing tasks including:
    - Loading cloth images
    - Extracting dimensions
    - Finding usable area
    - Managing model
    """

    # Magic numbers for cloth processing
    DEFAULT_WIDTH = 200  # Default cloth width in cm
    DEFAULT_HEIGHT = 300  # Default cloth height in cm
    IMAGE_SIZE = 256  # Image size for neural network
    EDGE_MARGIN = 5  # Margin from edges in cm (for clean cuts)

    def __init__(self, model_path: str = "models/cloth_model.pth"):
        self.model_path = model_path
        self.model = SimpleClothCNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"Cloth Processor initialized. Using device: {self.device}")

    def extract_dimensions_from_filename(self, filename: str) -> Tuple[float, float]:
        """
        Extract dimensions from filename format: cloth_widthxheight.jpg
        """
        match = re.search(r"(\d+)x(\d+)", filename)
        if match:
            width = float(match.group(1))
            height = float(match.group(2))
            logger.info(f"Extracted cloth dimensions from {filename}: {width}x{height}")
            return width, height
        else:
            logger.warning(
                f"No dimensions in {filename}, using defaults: {self.DEFAULT_WIDTH}x{self.DEFAULT_HEIGHT}"
            )
            return self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT

    def detect_cloth_area(self, image_path: str) -> Dict:
        """
        Detect the actual cloth area using simple computer vision.
        Returns the bounding box and mask of the cloth.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Simple cloth detection using color thresholding
        # This assumes cloth is different from background
        # You might need to adjust these values based on your images
        lower_bound = np.array([0, 0, 50])  # Adjust for your cloth
        upper_bound = np.array([180, 255, 255])

        # Create mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour (assumed to be the cloth)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            return {
                "bbox": (x, y, w, h),
                "mask": mask,
                "contour": largest_contour,
                "area": cv2.contourArea(largest_contour),
            }

        return None

    def process_image(self, image_path: str) -> Dict:
        """
        Process a single cloth image and extract all information.
        """
        logger.info(f"Processing cloth image: {image_path}")

        # Extract dimensions
        filename = os.path.basename(image_path)
        width, height = self.extract_dimensions_from_filename(filename)

        # Detect cloth area
        cloth_area = self.detect_cloth_area(image_path)

        # Load and preprocess image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.IMAGE_SIZE, self.IMAGE_SIZE))

        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)

        # Calculate usable area (considering margins)
        usable_width = width - 2 * self.EDGE_MARGIN
        usable_height = height - 2 * self.EDGE_MARGIN

        return {
            "filename": filename,
            "path": image_path,
            "dimensions": (width, height),
            "usable_dimensions": (usable_width, usable_height),
            "model_features": output["features"].cpu().numpy(),
            "model_dimensions": output["dimensions"].cpu().numpy(),
            "cloth_area": cloth_area,
            "original_image": img_rgb,
            "total_area": width * height,
            "usable_area": usable_width * usable_height,
        }

    def load_model(self) -> bool:
        """
        Load pre-trained model if it exists.
        """
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing cloth model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Cloth model loaded successfully!")
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
        logger.info(f"Cloth model saved to {self.model_path}")
