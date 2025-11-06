"""
Pattern Recognition Module

Handles garment pattern detection, classification, and feature extraction.
Processes both SVG panel images and traditional pattern images.
Balances simplicity with functionality.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import cairosvg
from io import BytesIO

from .config import IMAGENET_NORMALIZE, PATTERN, SYSTEM

# Setup logging
logging.basicConfig(
    level=getattr(logging, str(SYSTEM["LOG_LEVEL"])), format=str(SYSTEM["LOG_FORMAT"])
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
    key_points: Optional[List[Tuple[float, float]]] = None  # Important feature points


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


class PatternDataset(Dataset):
    """Custom dataset for loading pattern images."""

    def __init__(
        self, image_paths: List[str], transforms: transforms.Compose, pattern_types: List[str]
    ):
        self.image_paths = image_paths
        self.transforms = transforms
        self.pattern_types = pattern_types

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img_path = self.image_paths[idx]
        
        if img_path.lower().endswith(".svg"):
            png_data = cairosvg.svg2png(url=img_path)
            img = Image.open(BytesIO(png_data)).convert("RGB")
        else:
            img = Image.open(img_path).convert("RGB")

        # Extract label from path
        label_str = "other"
        for p_type in self.pattern_types:
            if p_type in img_path:
                label_str = p_type
                break
        label = self.pattern_types.index(label_str)

        # Extract dimensions from filename
        width, height = self._extract_dimensions_from_filename(os.path.basename(img_path))
        dims = torch.tensor([width, height], dtype=torch.float32)

        img_tensor = self.transforms(img)
        return img_tensor, label, dims

    def _extract_dimensions_from_filename(self, filename: str) -> Tuple[float, float]:
        match = re.search(r"(\d+)x(\d+)", filename)
        if match:
            return float(match.group(1)), float(match.group(2))

        # Return default/standard dimensions if not in filename
        return 20.0, 20.0


class PatternRecognitionModule:
    """
    Main class for pattern recognition that handles:
    1. Loading and preprocessing pattern images
    2. Extracting pattern shapes and dimensions
    3. Pattern type classification
    4. Model training and inference
    """

    def __init__(self, model_path: Optional[str] = None):
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

        # Data augmentation and normalization for training
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((PATTERN["IMAGE_SIZE"], PATTERN["IMAGE_SIZE"])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]
                ),
            ]
        )

        # Normalization for validation/testing
        self.val_transforms = transforms.Compose(
            [
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

    def extract_shape_from_svg(
        self, svg_path: str
    ) -> Dict[str, Union[np.ndarray, List[Tuple[float, float]]]]:
        """
        Extract pattern shape from SVG file generated by extract_panel_dimensions.py.
        Parses SVG path elements to get contour points.
        """
        try:
            # Parse SVG
            tree = ET.parse(svg_path)
            root = tree.getroot()

            # Find path elements
            path_elements = root.findall(".//{http://www.w3.org/2000/svg}path")
            if not path_elements:
                # Try without namespace
                path_elements = root.findall(".//path")

            if not path_elements:
                logger.warning(f"No path elements found in SVG: {svg_path}")
                return {"contour": np.array([]), "key_points": []}

            # Extract points from first path (simplified approach)
            path_data = path_elements[0].get("d", "")

            # Parse path data to extract coordinates
            points = []
            # Simple parsing - look for numbers in path string
            import re

            coords = re.findall(r"([0-9.]+)", path_data)

            # Group coordinates into pairs
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x, y = float(coords[i]), float(coords[i + 1])
                    points.append([x, y])

            if len(points) < 3:
                logger.warning(f"Insufficient points in SVG: {svg_path}")
                return {"contour": np.array([]), "key_points": []}

            # Convert to numpy array and reshape for OpenCV
            contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

            # Extract key points as corner points
            key_points = [(float(p[0]), float(p[1])) for p in points]

            return {"contour": contour, "key_points": key_points}

        except Exception as e:
            logger.error(f"Failed to parse SVG {svg_path}: {e}")
            return {"contour": np.array([]), "key_points": []}

    def extract_pattern_shape(
        self, image_path: str
    ) -> Dict[str, Union[np.ndarray, List[Tuple[float, float]]]]:
        """
        Extract the pattern shape from an image or SVG using computer vision.
        Returns contour and key points.
        """
        # Handle SVG files (from panel extraction)
        if image_path.lower().endswith(".svg"):
            return self.extract_shape_from_svg(image_path)

        # Handle raster images (PNG, JPG)
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
            logger.warning(
                f"No contours found in {image_path}, falling back to full image"
            )
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
        key_points: List[Tuple[float, float]] = []

        # Method 1: Use simplified polygon vertices
        if len(simplified) >= 3:
            key_points = []
            for pt in simplified:
                x = float(pt.flatten()[0])
                y = float(pt.flatten()[1])
                key_points.append((x, y))

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
                key_points = [
                    (float(corner.ravel()[0]), float(corner.ravel()[1]))
                    for corner in corners
                ]

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
            # Use PIL to convert numpy array to tensor properly
            from PIL import Image

            pil_img = Image.fromarray(img_rgb)
            transformed = self.val_transforms(pil_img)
            if not isinstance(transformed, torch.Tensor):
                transformed = torch.tensor(transformed)
            img_tensor = transformed.unsqueeze(0).to(self.device)

            # Run inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(img_tensor)

            # Get pattern type prediction
            type_scores = F.softmax(output["pattern_type"], dim=1)
            type_idx = int(torch.argmax(type_scores, dim=1).item())
            pattern_type = self.pattern_types[type_idx]
            confidence = float(type_scores[0, type_idx].item())

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
            if isinstance(contour, np.ndarray) and len(contour) > 0:
                width, height = self.estimate_dimensions(contour, pattern_type)
            else:
                # Fallback to standard dimensions
                standard = PATTERN["STANDARD_DIMENSIONS"].get(
                    pattern_type, PATTERN["STANDARD_DIMENSIONS"]["other"]
                )
                width, height = standard["width"], standard["height"]

        # Calculate area
        if isinstance(contour, np.ndarray) and len(contour) > 2:
            # Use contour area
            pixel_area = cv2.contourArea(contour)
            area = pixel_area * (PATTERN["PIXEL_TO_CM"] ** 2)
        else:
            # Fallback to rectangular area
            area = width * height

        # Ensure proper types for Pattern object
        contour_array = contour if isinstance(contour, np.ndarray) else np.array([])
        key_points_list = key_points if isinstance(key_points, list) else []

        # Create Pattern object
        pattern = Pattern(
            id=hash(image_path),
            name=name,
            pattern_type=pattern_type,
            width=width,
            height=height,
            area=area,
            contour=contour_array,
            confidence=confidence,
            key_points=key_points_list,
        )

        logger.info(
            f"Pattern detected: {pattern_type} ({confidence:.2f}), "
            f"{width:.1f}x{height:.1f} cm, Area: {area:.1f} cm²"
        )

        return pattern

    def save_model(self) -> None:
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

    def train(
        self,
        train_images: List[str],
        val_images: List[str],
        epochs: int = 10,
        batch_size: int = 16,
    ) -> Dict:
        """Train the pattern recognition model."""
        logger.info(
            f"Training pattern recognition model with {len(train_images)} train images and {len(val_images)} validation images"
        )

        # Create datasets and dataloaders
        train_dataset = PatternDataset(train_images, self.train_transforms, self.pattern_types)
        val_dataset = PatternDataset(val_images, self.val_transforms, self.pattern_types)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Loss functions and optimizer
        criterion_type = nn.CrossEntropyLoss()
        criterion_dims = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels, dims in train_loader:
                images, labels, dims = (
                    images.to(self.device),
                    labels.to(self.device),
                    dims.to(self.device),
                )

                optimizer.zero_grad()
                outputs = self.model(images)
                loss_type = criterion_type(outputs["pattern_type"], labels)
                loss_dims = criterion_dims(outputs["dimensions"], dims)
                loss = loss_type + loss_dims
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels, dims in val_loader:
                    images, labels, dims = (
                        images.to(self.device),
                        labels.to(self.device),
                        dims.to(self.device),
                    )
                    outputs = self.model(images)
                    loss_type = criterion_type(outputs["pattern_type"], labels)
                    loss_dims = criterion_dims(outputs["dimensions"], dims)
                    loss = loss_type + loss_dims
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs["pattern_type"].data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()

        return history

