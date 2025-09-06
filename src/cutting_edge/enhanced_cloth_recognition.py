"""
Enhanced Cloth Recognition Module
Handles cloth materials with varying shapes and dimensions.
Cloth can be rectangular, irregular, or have defects/holes.
"""

import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

from .simple_config import CLOTH, SYSTEM, IMAGENET_NORMALIZE

# Setup logging
logging.basicConfig(level=logging.INFO, format=SYSTEM["LOG_FORMAT"])
logger = logging.getLogger(__name__)


@dataclass
class ClothMaterial:
    """Data class for cloth material with variable shape"""

    id: int
    name: str
    cloth_type: str  # cotton, silk, etc.
    total_width: float  # Bounding box width in cm
    total_height: float  # Bounding box height in cm
    usable_area: float  # Actual usable area in cm²
    contour: np.ndarray  # Actual cloth shape (can be irregular)
    defects: List[np.ndarray]  # List of holes/defects in the cloth
    texture_features: Optional[np.ndarray] = None  # For material property detection


class ClothSegmenter(nn.Module):
    """
    U-Net based segmentation network to extract actual cloth shape.
    Handles irregular shapes and detects defects/unusable areas.
    """

    def __init__(self):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  # 512 + 512 from skip connection

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)  # 256 + 256 from skip connection

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)  # 128 + 128 from skip connection

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)  # 64 + 64 from skip connection

        # Output layers
        self.cloth_mask = nn.Conv2d(64, 1, kernel_size=1)  # Binary mask for cloth
        self.defect_mask = nn.Conv2d(64, 1, kernel_size=1)  # Binary mask for defects

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        """Convolutional block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output masks
        cloth_mask = torch.sigmoid(self.cloth_mask(dec1))
        defect_mask = torch.sigmoid(self.defect_mask(dec1))

        return {
            "cloth_mask": cloth_mask,
            "defect_mask": defect_mask,
            "features": bottleneck,
        }


class EnhancedClothProcessor:
    """
    Processes cloth images to extract actual usable areas.
    Handles irregular shapes, defects, and material properties.
    """

    def __init__(self, model_path: str = "models/cloth_segmenter.pth"):
        self.model_path = model_path
        self.model = ClothSegmenter()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and SYSTEM["USE_GPU"] else "cpu"
        )
        self.model.to(self.device)

        # Cloth types for classification
        self.cloth_types = CLOTH["TYPES"]

        logger.info(f"Enhanced Cloth Processor initialized. Device: {self.device}")

    def extract_cloth_shape(self, image_path: str) -> Dict:
        """
        Extract the actual cloth shape, handling irregular materials.
        Returns cloth contour and any defects/holes.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        h_orig, w_orig = img.shape[:2]

        # Method 1: Try U-Net segmentation (if model is trained)
        try:
            # Prepare image for U-Net
            img_resized = cv2.resize(img, (CLOTH["IMAGE_SIZE"], CLOTH["IMAGE_SIZE"]))
            img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # Get segmentation masks
            self.model.eval()
            with torch.no_grad():
                output = self.model(img_tensor)

            # Convert masks to numpy
            cloth_mask = output["cloth_mask"].cpu().numpy()[0, 0]
            defect_mask = output["defect_mask"].cpu().numpy()[0, 0]

            # Resize masks back to original size
            cloth_mask = cv2.resize(
                (cloth_mask * 255).astype(np.uint8), (w_orig, h_orig)
            )
            defect_mask = cv2.resize(
                (defect_mask * 255).astype(np.uint8), (w_orig, h_orig)
            )

            # Threshold to get binary masks
            _, cloth_binary = cv2.threshold(cloth_mask, 127, 255, cv2.THRESH_BINARY)
            _, defect_binary = cv2.threshold(defect_mask, 127, 255, cv2.THRESH_BINARY)

        except Exception as e:
            logger.warning(f"U-Net segmentation failed, using color-based method: {e}")
            # Method 2: Fallback to color-based detection
            cloth_binary, defect_binary = self.color_based_segmentation(img)

        # Extract contours
        cloth_contours, _ = cv2.findContours(
            cloth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        defect_contours, _ = cv2.findContours(
            defect_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get main cloth contour (largest)
        if cloth_contours:
            main_contour = max(cloth_contours, key=cv2.contourArea)

            # Filter defects (keep only those inside cloth)
            valid_defects = []
            for defect in defect_contours:
                # Check if defect is inside cloth contour
                M = cv2.moments(defect)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if cv2.pointPolygonTest(main_contour, (cx, cy), False) >= 0:
                        valid_defects.append(defect)
        else:
            # No cloth detected, assume entire image is cloth
            main_contour = np.array(
                [[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]]
            )
            valid_defects = []

        return {
            "main_contour": main_contour,
            "defects": valid_defects,
            "cloth_mask": cloth_binary,
            "defect_mask": defect_binary,
        }

    def color_based_segmentation(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback method using color-based segmentation.
        Returns cloth mask and defect mask.
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detect cloth (non-background)
        lower_bound = np.array(CLOTH["HSV_LOWER"])
        upper_bound = np.array(CLOTH["HSV_UPPER"])
        cloth_mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Clean up with morphology
        kernel = np.ones(
            (CLOTH["MORPH_KERNEL_SIZE"], CLOTH["MORPH_KERNEL_SIZE"]), np.uint8
        )
        cloth_mask = cv2.morphologyEx(
            cloth_mask, cv2.MORPH_CLOSE, kernel, iterations=CLOTH["MORPH_ITERATIONS"]
        )
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel)

        # Detect defects (very dark or very bright spots within cloth)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, dark_spots = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        _, bright_spots = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
        defect_mask = cv2.bitwise_or(dark_spots, bright_spots)
        defect_mask = cv2.bitwise_and(
            defect_mask, cloth_mask
        )  # Only defects within cloth

        return cloth_mask, defect_mask

    def calculate_usable_area(
        self,
        contour: np.ndarray,
        defects: List[np.ndarray],
        scale_x: float,
        scale_y: float,
    ) -> float:
        """
        Calculate actual usable area considering defects.
        Returns area in cm².
        """
        # Main cloth area
        main_area = (
            cv2.contourArea(contour) * scale_x * scale_y * (CLOTH["PIXEL_TO_CM"] ** 2)
        )

        # Subtract defect areas
        defect_area = 0
        for defect in defects:
            defect_area += (
                cv2.contourArea(defect)
                * scale_x
                * scale_y
                * (CLOTH["PIXEL_TO_CM"] ** 2)
            )

        # Consider edge margins
        # Create polygon and buffer inward by margin
        try:
            poly = Polygon(contour.reshape(-1, 2))
            margin_cm = CLOTH["EDGE_MARGIN"]
            margin_pixels = margin_cm / CLOTH["PIXEL_TO_CM"]
            buffered = poly.buffer(-margin_pixels)  # Negative buffer = inward
            if buffered.area > 0:
                usable_area = (
                    buffered.area * scale_x * scale_y * (CLOTH["PIXEL_TO_CM"] ** 2)
                    - defect_area
                )
            else:
                usable_area = main_area * 0.9 - defect_area  # Fallback: use 90% of area
        except:
            usable_area = main_area * 0.9 - defect_area  # Fallback

        return max(0, usable_area)

    def process_cloth(self, image_path: str) -> ClothMaterial:
        """
        Process cloth image to extract shape, defects, and properties.
        """
        logger.info(f"Processing cloth: {image_path}")

        # Extract dimensions from filename
        filename = os.path.basename(image_path)
        match = re.search(r"(\d+)x(\d+)", filename)

        # Load image and get shape info
        img = cv2.imread(image_path)
        shape_info = self.extract_cloth_shape(image_path)

        # Get bounding box of main contour
        x, y, w, h = cv2.boundingRect(shape_info["main_contour"])

        # Determine dimensions and scale
        if match:
            # Use dimensions from filename
            total_width = float(match.group(1))
            total_height = float(match.group(2))
            # Calculate scale from image to real dimensions
            scale_x = total_width / w if w > 0 else 1
            scale_y = total_height / h if h > 0 else 1
            logger.info(
                f"Using dimensions from filename: {total_width}x{total_height} cm"
            )
        else:
            # Use default dimensions
            total_width = CLOTH["DEFAULT_WIDTH"]
            total_height = CLOTH["DEFAULT_HEIGHT"]
            scale_x = total_width / img.shape[1]
            scale_y = total_height / img.shape[0]
            logger.info(f"Using default dimensions: {total_width}x{total_height} cm")

        # Calculate usable area
        usable_area = self.calculate_usable_area(
            shape_info["main_contour"], shape_info["defects"], scale_x, scale_y
        )

        # Detect cloth type (simplified - could use neural network)
        cloth_type = self.detect_cloth_type(img, shape_info["main_contour"])

        # Extract texture features for material properties
        texture_features = self.extract_texture_features(img, shape_info["cloth_mask"])

        # Create ClothMaterial object
        cloth = ClothMaterial(
            id=hash(filename),
            name=os.path.splitext(filename)[0],
            cloth_type=cloth_type,
            total_width=total_width,
            total_height=total_height,
            usable_area=usable_area,
            contour=shape_info["main_contour"],
            defects=shape_info["defects"],
            texture_features=texture_features,
        )

        logger.info(
            f"Cloth: {cloth_type}, Size: {total_width:.1f}x{total_height:.1f} cm"
        )
        logger.info(
            f"Usable area: {usable_area:.1f} cm² ({len(shape_info['defects'])} defects found)"
        )

        return cloth

    def detect_cloth_type(self, img: np.ndarray, contour: np.ndarray) -> str:
        """
        Simple cloth type detection based on color and texture.
        In a real system, this would use a trained classifier.
        """
        # Create mask from contour
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Calculate average color in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]

        # Simple heuristic based on color
        hue = mean_hsv[0]
        saturation = mean_hsv[1]
        value = mean_hsv[2]

        # Very simple classification (would be ML model in practice)
        if saturation < 30:  # Low saturation
            if value > 200:
                return "cotton"  # Light, neutral
            else:
                return "wool"  # Dark, neutral
        else:
            if hue < 30 or hue > 150:  # Reddish
                return "silk"
            else:
                return "polyester"  # Other colors

    def extract_texture_features(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract texture features that might affect cutting.
        E.g., grain direction, stretchiness indicators.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gabor filters to detect texture directionality
        features = []

        # Multiple orientations
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel(
                ksize=(31, 31), sigma=4.0, theta=theta, lambd=10.0, gamma=0.5
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)

            # Calculate mean response in cloth area
            mean_response = cv2.mean(filtered, mask=mask)[0]
            features.append(mean_response)

        return np.array(features)

    def visualize_cloth_analysis(self, cloth: ClothMaterial, output_path: str):
        """
        Visualize the cloth shape and defects for verification.
        """
        # Create blank canvas
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Draw cloth shape
        ax1.set_title(f"Cloth Shape: {cloth.cloth_type}")
        ax1.set_aspect("equal")

        # Draw main contour
        if len(cloth.contour) > 0:
            contour = cloth.contour.squeeze()
            ax1.fill(
                contour[:, 0],
                contour[:, 1],
                color="lightblue",
                alpha=0.7,
                label="Usable cloth",
            )
            ax1.plot(contour[:, 0], contour[:, 1], "b-", linewidth=2)

        # Draw defects
        for i, defect in enumerate(cloth.defects):
            defect = defect.squeeze()
            if len(defect) > 2:
                ax1.fill(
                    defect[:, 0],
                    defect[:, 1],
                    color="red",
                    alpha=0.8,
                    label="Defect" if i == 0 else "",
                )

        ax1.legend()
        ax1.set_xlabel("Width (pixels)")
        ax1.set_ylabel("Height (pixels)")

        # Show dimensions and area info
        ax2.axis("off")
        info_text = f"""Cloth Information:
        
Type: {cloth.cloth_type}
Dimensions: {cloth.total_width:.1f} x {cloth.total_height:.1f} cm
Usable Area: {cloth.usable_area:.1f} cm²
Efficiency: {(cloth.usable_area / (cloth.total_width * cloth.total_height) * 100):.1f}%
Defects Found: {len(cloth.defects)}
Edge Margin: {CLOTH["EDGE_MARGIN"]} cm"""

        ax2.text(
            0.1,
            0.5,
            info_text,
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="center",
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Cloth analysis visualization saved to {output_path}")

    def save_model(self):
        """Save the cloth segmentation model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "cloth_types": self.cloth_types,
            },
            self.model_path,
        )
        logger.info(f"Cloth segmenter saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load the cloth segmentation model"""
        if os.path.exists(self.model_path):
            logger.info(f"Loading cloth segmenter from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            if "cloth_types" in checkpoint:
                self.cloth_types = checkpoint["cloth_types"]

            logger.info("Cloth segmenter loaded successfully")
            return True
        return False
