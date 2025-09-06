"""
Enhanced Pattern Recognition Module
Handles fixed pattern shapes with varying dimensions.
Patterns are templates (shirt, pants, etc.) that need to be cut from cloth.
"""

import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging
from dataclasses import dataclass
import torchvision.transforms as transforms

from .simple_config import PATTERN, TRAINING, IMAGENET_NORMALIZE, SYSTEM

# Setup logging
logging.basicConfig(level=logging.INFO, format=SYSTEM["LOG_FORMAT"])
logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Data class for a fixed pattern shape"""

    id: int
    name: str
    pattern_type: str  # shirt, pants, dress, etc.
    width: float  # in cm
    height: float  # in cm
    contour: np.ndarray  # Shape outline
    area: float  # Total area needed
    key_points: List[Tuple[float, float]] = None  # Important points (corners, curves)


class PatternRecognizer(nn.Module):
    """
    Neural network to recognize pattern types and extract key features.
    Patterns are fixed shapes, we just need to identify type and scale.
    """

    def __init__(self, num_pattern_types: int = len(PATTERN["TYPES"])):
        super().__init__()

        # Convolutional backbone for feature extraction
        self.backbone = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # Pattern type classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_pattern_types),
        )

        # Dimension estimator (scale factor from standard template)
        self.scale_estimator = nn.Sequential(
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # width_scale, height_scale
        )

    def forward(self, x):
        features = self.backbone(x)
        features_flat = features.view(features.size(0), -1)

        pattern_type = self.classifier(features_flat)
        scale_factors = self.scale_estimator(features_flat)

        return {
            "pattern_type": pattern_type,
            "scale_factors": scale_factors,
            "features": features_flat,
        }


class EnhancedPatternProcessor:
    """
    Processes pattern images to extract fixed shapes and their dimensions.
    Patterns are templates that will be cut from cloth material.
    """

    # Standard pattern dimensions (templates) in cm
    STANDARD_PATTERNS = {
        "shirt": {"width": 60, "height": 80},
        "pants": {"width": 45, "height": 110},
        "dress": {"width": 55, "height": 120},
        "sleeve": {"width": 25, "height": 60},
        "collar": {"width": 40, "height": 15},
        "other": {"width": 50, "height": 50},
    }

    def __init__(self, model_path: str = "models/pattern_recognizer.pth"):
        self.model_path = model_path
        self.model = PatternRecognizer()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and SYSTEM["USE_GPU"] else "cpu"
        )
        self.model.to(self.device)

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((PATTERN["IMAGE_SIZE"], PATTERN["IMAGE_SIZE"])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]
                ),
            ]
        )

        self.pattern_types = PATTERN["TYPES"]
        logger.info(f"Pattern Processor initialized. Device: {self.device}")

    def extract_pattern_shape(self, image_path: str) -> np.ndarray:
        """
        Extract the actual shape/contour of the pattern.
        This is the fixed shape that needs to be cut from cloth.
        """
        img = cv2.imread(image_path)
        if img is None:
            return np.array([])

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use OTSU thresholding to separate pattern from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean up with morphology
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find the main contour
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.array([])

        # Get the largest contour (the pattern shape)
        main_contour = max(contours, key=cv2.contourArea)

        # Simplify the contour to reduce points
        perimeter = cv2.arcLength(main_contour, True)
        epsilon = 0.01 * perimeter  # 1% of perimeter
        simplified = cv2.approxPolyDP(main_contour, epsilon, True)

        return simplified

    def extract_key_points(self, contour: np.ndarray) -> List[Tuple[float, float]]:
        """
        Extract key points from pattern shape (corners, curves).
        These help in optimal placement and rotation.
        """
        if len(contour) < 4:
            return []

        # Find convex hull for outer boundary
        hull = cv2.convexHull(contour)

        # Find corners using Harris corner detection on the contour mask
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        corners = cv2.goodFeaturesToTrack(
            mask, maxCorners=20, qualityLevel=0.01, minDistance=20
        )

        key_points = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                key_points.append((float(x), float(y)))

        return key_points

    def process_pattern(self, image_path: str) -> Pattern:
        """
        Process a pattern image to extract its fixed shape and dimensions.
        """
        logger.info(f"Processing pattern: {image_path}")

        # Extract actual dimensions from filename
        filename = os.path.basename(image_path)
        match = re.search(r"(\d+)x(\d+)", filename)

        # Load and process image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get pattern type from neural network
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)

        # Determine pattern type
        pattern_probs = F.softmax(output["pattern_type"], dim=1)
        pattern_idx = torch.argmax(pattern_probs, dim=1).item()
        pattern_type = self.pattern_types[pattern_idx]
        confidence = pattern_probs[0, pattern_idx].item()

        # Get dimensions
        if match:
            # Use dimensions from filename
            width = float(match.group(1))
            height = float(match.group(2))
            logger.info(f"Using dimensions from filename: {width}x{height} cm")
        else:
            # Use standard dimensions with predicted scale
            scale_factors = output["scale_factors"].cpu().numpy()[0]
            standard = self.STANDARD_PATTERNS[pattern_type]
            width = standard["width"] * max(
                0.5, min(2.0, scale_factors[0])
            )  # Limit scale 0.5-2x
            height = standard["height"] * max(0.5, min(2.0, scale_factors[1]))
            logger.info(
                f"Estimated dimensions for {pattern_type}: {width:.1f}x{height:.1f} cm"
            )

        # Extract the fixed shape contour
        contour = self.extract_pattern_shape(image_path)

        # Calculate area
        if len(contour) > 0:
            # Scale contour to match dimensions
            rect = cv2.boundingRect(contour)
            scale_x = width / rect[2] if rect[2] > 0 else 1
            scale_y = height / rect[3] if rect[3] > 0 else 1
            area = (
                cv2.contourArea(contour)
                * scale_x
                * scale_y
                * (PATTERN["PIXEL_TO_CM"] ** 2)
            )
        else:
            area = width * height

        # Extract key points
        key_points = self.extract_key_points(contour)

        # Create Pattern object
        pattern = Pattern(
            id=hash(filename),  # Unique ID based on filename
            name=os.path.splitext(filename)[0],
            pattern_type=pattern_type,
            width=width,
            height=height,
            contour=contour,
            area=area,
            key_points=key_points,
        )

        logger.info(
            f"Pattern: {pattern_type}, Size: {width:.1f}x{height:.1f} cm, Area: {area:.1f} cm², Confidence: {confidence:.2f}"
        )

        return pattern

    def save_model(self):
        """Save the pattern recognizer model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "pattern_types": self.pattern_types,
                "standard_patterns": self.STANDARD_PATTERNS,
            },
            self.model_path,
        )
        logger.info(f"Pattern recognizer saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load the pattern recognizer model"""
        if os.path.exists(self.model_path):
            logger.info(f"Loading pattern recognizer from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            if "pattern_types" in checkpoint:
                self.pattern_types = checkpoint["pattern_types"]
            if "standard_patterns" in checkpoint:
                self.STANDARD_PATTERNS = checkpoint["standard_patterns"]

            logger.info("Pattern recognizer loaded successfully")
            return True
        return False
