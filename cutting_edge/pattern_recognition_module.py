"""
Pattern Recognition Module

Handles pattern detection, classification, and feature extraction.
Balances simplicity with functionality.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from .config import IMAGENET_NORMALIZE, PATTERN, SYSTEM

# Setup logging
logging.basicConfig(
    level=getattr(logging, SYSTEM["LOG_LEVEL"]), format=SYSTEM["LOG_FORMAT"]
)
logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """
    Data class to represent a garment pattern with all its properties.
    This is the unified format for patterns throughout the system.
    """

    id: int  # Unique identifier
    name: str  # Pattern name (typically from filename)
    pattern_type: str  # Type (shirt, pants, etc.)
    width: float  # Width in cm
    height: float  # Height in cm
    area: float  # Area in cm²
    contour: np.ndarray  # Shape outline points
    confidence: float = 1.0  # Confidence in classification
    key_points: List[Tuple[float, float]] = None  # Important feature points


class PatternRecognizer(nn.Module):
    """
    Neural network that handles pattern classification and dimension estimation.
    Uses either a simple CNN or pretrained backbone depending on config.
    """

    def __init__(self, num_classes: int = len(PATTERN["TYPES"])):
        super().__init__()
        self.backbone_type = PATTERN["BACKBONE"]
        self.feature_dim = PATTERN["FEATURE_DIM"]
        self.num_classes = num_classes

        # Initialize neural network
        self.backbone, self.feature_dim = self._create_backbone()

        # Pattern type classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # Dimension estimator (width/height)
        self.dimension_estimator = nn.Sequential(
            nn.Linear(self.feature_dim, PATTERN["ESTIMATOR_HIDDEN_DIM"]),
            nn.ReLU(),
            nn.Linear(PATTERN["ESTIMATOR_HIDDEN_DIM"], 2),
        )

    def _create_backbone(self) -> Tuple[nn.Module, int]:
        """Create the appropriate feature extraction backbone based on config."""
        if self.backbone_type == "simple":
            # Simple CNN backbone for learning purposes
            backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            feature_dim = 128 * 4 * 4

        elif self.backbone_type == "resnet18":
            # ResNet18 backbone - good balance of performance and size
            model = models.resnet18(weights="DEFAULT")
            backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
            feature_dim = 512

        elif self.backbone_type == "efficientnet-b0":
            # EfficientNet-B0 - lightweight but powerful
            model = models.efficientnet_b0(weights="DEFAULT")
            backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
            feature_dim = 1280

        else:
            raise ValueError(f"Unknown backbone type: {self.backbone_type}")

        return backbone, feature_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the neural network."""
        features = self.backbone(x)

        # Pattern type classification
        pattern_type = self.classifier(features)

        # Dimension estimation
        dimensions = self.dimension_estimator(features)

        return {
            "features": features,
            "pattern_type": pattern_type,
            "dimensions": dimensions,
        }


class PatternRecognitionModule:
    """
    Main class for pattern recognition that handles:
    1. Loading and preprocessing pattern images
    2. Extracting pattern shapes and dimensions
    3. Pattern type classification
    4. Model training and inference
    """

    def __init__(self, model_path: str = None):
        """Initialize the pattern recognition module."""
        if model_path is None:
            model_path = os.path.join(
                SYSTEM["BASE_DIR"], SYSTEM["MODELS_DIR"], "pattern_model.pth"
            )

        self.model_path = model_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and SYSTEM["USE_GPU"] else "cpu"
        )

        # Create neural network
        self.model = PatternRecognizer()
        self.model.to(self.device)

        # Image preprocessing transforms
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((PATTERN["IMAGE_SIZE"], PATTERN["IMAGE_SIZE"])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]
                ),
            ]
        )

        # Pattern type mapping
        self.pattern_types = PATTERN["TYPES"]

        logger.info(
            f"Pattern Recognition Module initialized. Using device: {self.device}"
        )
        logger.info(f"Using backbone: {PATTERN['BACKBONE']}")

    def extract_pattern_shape(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        Extract the pattern shape from an image using computer vision.
        Returns contour and key points.
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return {"contour": np.array([]), "key_points": []}

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to separate pattern from background
        # Try adaptive thresholding first
        try:
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                PATTERN["ADAPTIVE_THRESH_BLOCK_SIZE"],
                PATTERN["ADAPTIVE_THRESH_C"],
            )
        except Exception:
            # Fallback to regular thresholding
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        # Clean up with morphology operations
        kernel = np.ones(PATTERN["MORPH_KERNEL_SIZE"], np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # No contours found
        if not contours:
            logger.warning(f"No contours found in {image_path}, falling back to full image")
            h, w = img.shape[:2]
            main_contour = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
        else:
            # Get the largest contour (assumed to be the pattern)
            main_contour = max(contours, key=cv2.contourArea)

        # Simplify the contour
        perimeter = cv2.arcLength(main_contour, True)
        epsilon = PATTERN["CONTOUR_SIMPLIFICATION"] * perimeter
        simplified = cv2.approxPolyDP(main_contour, epsilon, True)

        # Extract key points (corners or significant points)
        key_points = []

        # Method 1: Use simplified polygon vertices
        if len(simplified) >= 3:
            key_points = [tuple(pt[0]) for pt in simplified]

        # Method 2: Fallback to corner detection if not enough points
        if len(key_points) < 3:
            # Create mask with the pattern
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [main_contour], 0, 255, -1)

            # Detect corners
            corners = cv2.goodFeaturesToTrack(
                mask,
                maxCorners=PATTERN["MAX_CORNERS"],
                qualityLevel=PATTERN["CORNER_QUALITY_LEVEL"],
                minDistance=PATTERN["CORNER_MIN_DISTANCE"],
            )
            if corners is not None:
                key_points = [tuple(corner.ravel()) for corner in corners]

        return {"contour": simplified, "key_points": key_points}

    def extract_dimensions_from_filename(
        self, filename: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract width and height from filename if available.
        Format: pattern_50x80.png -> 50cm x 80cm
        """
        # Try to find a pattern like "50x80" in the filename
        match = re.search(r"(\d+)x(\d+)", filename)
        if match:
            width = float(match.group(1))
            height = float(match.group(2))
            logger.info(f"Extracted dimensions from filename: {width}x{height} cm")
            return width, height
        return None, None

    def estimate_dimensions(
        self, contour: np.ndarray, pattern_type: str
    ) -> Tuple[float, float]:
        """
        Estimate pattern dimensions based on contour and/or neural network predictions.
        Falls back to standard dimensions for the pattern type.
        """
        # Default to standard dimensions for this pattern type
        standard = PATTERN["STANDARD_DIMENSIONS"].get(
            pattern_type, PATTERN["STANDARD_DIMENSIONS"]["other"]
        )

        if len(contour) > 2:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Convert from pixels to cm
            pixel_to_cm = PATTERN["PIXEL_TO_CM"]
            width_cm = w * pixel_to_cm
            height_cm = h * pixel_to_cm

            # If the dimensions seem reasonable, use them
            if width_cm > 5 and height_cm > 5:
                return width_cm, height_cm

        # Fallback to standard dimensions
        return standard["width"], standard["height"]

    def process_image(self, image_path: str) -> Pattern:
        """
        Process a pattern image and extract all relevant information.
        This is the main entry point for pattern recognition.
        """
        logger.info(f"Processing pattern image: {image_path}")

        # Extract pattern shape
        shape_info = self.extract_pattern_shape(image_path)
        contour = shape_info["contour"]
        key_points = shape_info["key_points"]

        # Get filename and try to extract dimensions
        filename = os.path.basename(image_path)
        name = os.path.splitext(filename)[0]
        width, height = self.extract_dimensions_from_filename(filename)

        # Process with neural network if available
        pattern_type = "other"  # Default
        confidence = 1.0

        # Load the image for neural network
        img = cv2.imread(image_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self.transforms(img_rgb).unsqueeze(0).to(self.device)

            # Run inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(img_tensor)

            # Get pattern type prediction
            type_scores = F.softmax(output["pattern_type"], dim=1)
            type_idx = torch.argmax(type_scores, dim=1).item()
            pattern_type = self.pattern_types[type_idx]
            confidence = type_scores[0, type_idx].item()

            # Use network's dimension prediction if no dimensions in filename
            if width is None or height is None:
                # Get scale factors
                scale_factors = output["dimensions"].cpu().numpy()[0]

                # Get base dimensions for the predicted type
                standard = PATTERN["STANDARD_DIMENSIONS"].get(
                    pattern_type, PATTERN["STANDARD_DIMENSIONS"]["other"]
                )

                # Apply scale factors (clipped to reasonable range)
                scale_x = max(0.5, min(2.0, 1.0 + scale_factors[0]))
                scale_y = max(0.5, min(2.0, 1.0 + scale_factors[1]))

                width = standard["width"] * scale_x
                height = standard["height"] * scale_y

        # If still no dimensions, try to estimate from contour
        if width is None or height is None:
            width, height = self.estimate_dimensions(contour, pattern_type)

        # Calculate area
        if len(contour) > 2:
            # Use contour area
            pixel_area = cv2.contourArea(contour)
            area = pixel_area * (PATTERN["PIXEL_TO_CM"] ** 2)
        else:
            # Fallback to rectangular area
            area = width * height

        # Create Pattern object
        pattern = Pattern(
            id=hash(image_path),
            name=name,
            pattern_type=pattern_type,
            width=width,
            height=height,
            area=area,
            contour=contour,
            confidence=confidence,
            key_points=key_points,
        )

        logger.info(
            f"Pattern detected: {pattern_type} ({confidence:.2f}), "
            f"{width:.1f}x{height:.1f} cm, Area: {area:.1f} cm²"
        )

        return pattern

    def save_model(self):
        """Save the pattern recognition model."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "backbone_type": self.model.backbone_type,
                "feature_dim": self.model.feature_dim,
                "num_classes": self.model.num_classes,
                "pattern_types": self.pattern_types,
            },
            self.model_path,
        )
        logger.info(f"Pattern recognition model saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load the pattern recognition model."""
        if not os.path.exists(self.model_path):
            logger.warning(f"No model found at {self.model_path}")
            return False

        try:
            logger.info(f"Loading pattern recognition model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Load pattern types if available
            if "pattern_types" in checkpoint:
                self.pattern_types = checkpoint["pattern_types"]

            logger.info("Pattern recognition model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def train(self, train_images: List[str], epochs: int = 10) -> Dict:
        """Train the pattern recognition model."""
        logger.info(
            f"Training pattern recognition model with {len(train_images)} images"
        )

        # TODO: Implement proper training
        # For now, just save the model
        self.save_model()

        return {"status": "success", "message": "Model saved", "epochs_completed": 0}
