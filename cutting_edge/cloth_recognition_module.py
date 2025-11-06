"""
Cloth Recognition Module

Handles cloth material detection, segmentation, and property analysis.
Balances simplicity with functionality.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .config import CLOTH, SYSTEM

# Setup logging
logging.basicConfig(
    level=getattr(logging, SYSTEM["LOG_LEVEL"]), format=SYSTEM["LOG_FORMAT"]
)
logger = logging.getLogger(__name__)


@dataclass
class ClothMaterial:
    """
    Data class to represent cloth material with all its properties.
    This is the unified format for cloth materials throughout the system.
    """

    id: int  # Unique identifier
    name: str  # Material name (typically from filename)
    cloth_type: str  # Material type (cotton, silk, etc.)
    width: float  # Width in cm
    height: float  # Height in cm
    total_area: float  # Total area in cm²
    usable_area: float  # Usable area in cm² (excluding defects and margins)
    contour: np.ndarray  # Cloth boundary points
    defects: List[np.ndarray] = None  # Defect areas (holes, stains, etc.)
    material_properties: Dict = None  # Additional properties (stretch, grain, etc.)


class UNetSegmenter(nn.Module):
    """
    U-Net model for semantic segmentation of cloth images.
    Used when CLOTH["USE_UNET"] is True for more accurate segmentation.
    """

    def __init__(self, in_channels=3, out_channels=2):
        """Initialize U-Net with specified input/output channels."""
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bridge
        self.bridge = self._conv_block(512, 1024)

        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(1024, 512)  # 512 + 512 from skip connection

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(512, 256)  # 256 + 256 from skip connection

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)  # 128 + 128 from skip connection

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(128, 64)  # 64 + 64 from skip connection

        # Output layers
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_channels, out_channels):
        """Convolutional block with batch normalization."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass through U-Net architecture."""
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bridge
        bridge = self.bridge(self.pool(enc4))

        # Decoder with skip connections
        up1 = self.up1(bridge)
        dec1 = self.dec1(torch.cat([up1, enc4], dim=1))

        up2 = self.up2(dec1)
        dec2 = self.dec2(torch.cat([up2, enc3], dim=1))

        up3 = self.up3(dec2)
        dec3 = self.dec3(torch.cat([up3, enc2], dim=1))

        up4 = self.up4(dec3)
        dec4 = self.dec4(torch.cat([up4, enc1], dim=1))

        # Output segmentation masks
        output = self.output(dec4)

        return output


class ClothDataset(Dataset):
    """Custom dataset for loading cloth images and masks."""

    def __init__(self, image_paths: List[str], transforms):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = img_path.replace("images", "masks").replace(".jpg", ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return img, mask


class ClothRecognitionModule:
    """
    Main class for cloth recognition that handles:
    1. Loading and preprocessing cloth images
    2. Extracting cloth boundaries and defects
    3. Material type classification
    4. Model training and inference
    """

    def __init__(self, model_path: str = None):
        """Initialize the cloth recognition module."""
        if model_path is None:
            model_path = os.path.join(
                SYSTEM["BASE_DIR"], SYSTEM["MODELS_DIR"], "cloth_model.pth"
            )

        self.model_path = model_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and SYSTEM["USE_GPU"] else "cpu"
        )

        # Create segmentation model if configured to use U-Net
        self.use_unet = CLOTH["USE_UNET"]
        if self.use_unet:
            self.model = UNetSegmenter(
                in_channels=CLOTH["UNET_IN_CHANNELS"],
                out_channels=CLOTH["UNET_OUT_CHANNELS"],
            )
            self.model.to(self.device)

        # Material type mapping
        self.cloth_types = CLOTH["TYPES"]

        logger.info(
            f"Cloth Recognition Module initialized. Using device: {self.device}"
        )
        logger.info(f"Using U-Net segmentation: {self.use_unet}")

    def extract_dimensions_from_filename(
        self, filename: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract width and height from filename if available.
        Format: cloth_200x300.jpg -> interpret as pixels, convert to cm
        """
        # Try to find a pattern like "200x300" in the filename
        match = re.search(r"(\d+)x(\d+)", filename)
        if match:
            width_px = float(match.group(1))
            height_px = float(match.group(2))

            # If dimensions are large (>400), they're likely pixels not cm
            # Standard fabric width is 150cm (60 inches)
            if width_px > 400 or height_px > 400:
                # Convert pixels to cm using standard DPI
                # Assuming 100 DPI for fabric images
                width = width_px * 2.54 / 100  # inches to cm
                height = height_px * 2.54 / 100
                logger.info(
                    f"Converted pixel dimensions {width_px}x{height_px}px to {width:.1f}x{height:.1f} cm"
                )
            else:
                # Already in cm
                width = width_px
                height = height_px
                logger.info(f"Extracted dimensions from filename: {width}x{height} cm")
            return width, height
        return None, None

    def extract_cloth_shape(self, image_path: str) -> Dict:
        """
        Extract cloth shape and defects from an image.
        Uses either U-Net or color-based segmentation.
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return {
                "contour": np.array([]),
                "defects": [],
                "cloth_mask": None,
                "defect_mask": None,
            }

        h, w = img.shape[:2]

        if self.use_unet and hasattr(self, "model"):
            # U-Net segmentation approach
            try:
                # Prepare image for U-Net
                img_resized = cv2.resize(
                    img, (CLOTH["IMAGE_SIZE"], CLOTH["IMAGE_SIZE"])
                )
                img_tensor = (
                    torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
                )
                img_tensor = img_tensor.unsqueeze(0).to(self.device)

                # Get segmentation masks
                self.model.eval()
                with torch.no_grad():
                    output = self.model(img_tensor)

                # Convert output to masks (cloth and defects)
                output = F.softmax(output, dim=1)
                cloth_mask = output[:, 0].cpu().numpy()[0]
                defect_mask = output[:, 1].cpu().numpy()[0]

                # Resize masks to original image size
                cloth_mask = cv2.resize((cloth_mask * 255).astype(np.uint8), (w, h))
                defect_mask = cv2.resize((defect_mask * 255).astype(np.uint8), (w, h))

                # Threshold to get binary masks
                _, cloth_binary = cv2.threshold(
                    cloth_mask, CLOTH["THRESHOLD_VALUE"], 255, cv2.THRESH_BINARY
                )
                _, defect_binary = cv2.threshold(
                    defect_mask, CLOTH["THRESHOLD_VALUE"], 255, cv2.THRESH_BINARY
                )

            except Exception as e:
                logger.warning(
                    f"U-Net segmentation failed: {e}, falling back to color-based segmentation"
                )
                cloth_binary, defect_binary = self.color_based_segmentation(img)
        else:
            # Color-based segmentation approach
            cloth_binary, defect_binary = self.color_based_segmentation(img)

        # Extract contours
        contours, _ = cv2.findContours(
            cloth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        defect_contours, _ = cv2.findContours(
            defect_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get main cloth contour
        if contours:
            main_contour = max(contours, key=cv2.contourArea)

            # Filter defects (keep only those inside the cloth)
            valid_defects = []
            for defect in defect_contours:
                if (
                    cv2.contourArea(defect) > CLOTH["MIN_DEFECT_AREA"]
                ):  # Ignore tiny defects
                    # Check if defect is inside cloth
                    M = cv2.moments(defect)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if cv2.pointPolygonTest(main_contour, (cx, cy), False) >= 0:
                            valid_defects.append(defect)
        else:
            # If no contour found, assume the whole image is cloth
            main_contour = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
            valid_defects = []

        return {
            "contour": main_contour,
            "defects": valid_defects,
            "cloth_mask": cloth_binary,
            "defect_mask": defect_binary,
        }

    def color_based_segmentation(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple color-based segmentation for cloth and defects.
        Used as fallback or when U-Net is not configured.
        """
        # Convert to HSV for better color-based segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create cloth mask based on HSV ranges
        lower = np.array(CLOTH["HSV_LOWER"])
        upper = np.array(CLOTH["HSV_UPPER"])
        cloth_mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations to clean up
        kernel = np.ones(
            (CLOTH["MORPH_KERNEL_SIZE"], CLOTH["MORPH_KERNEL_SIZE"]), np.uint8
        )
        cloth_mask = cv2.morphologyEx(
            cloth_mask, cv2.MORPH_CLOSE, kernel, iterations=CLOTH["MORPH_ITERATIONS"]
        )
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel)

        # Simple defect detection (very dark or very bright spots)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, dark_spots = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        _, bright_spots = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

        defect_mask = cv2.bitwise_or(dark_spots, bright_spots)
        defect_mask = cv2.bitwise_and(
            defect_mask, cloth_mask
        )  # Only defects within cloth

        return cloth_mask, defect_mask

    def calculate_areas(
        self,
        contour: np.ndarray,
        defects: List[np.ndarray],
        width: float,
        height: float,
    ) -> Tuple[float, float]:
        """
        Calculate total and usable areas for cloth.
        Accounts for defects and edge margins.
        """
        # Check if contour is valid
        if contour is None or len(contour) == 0:
            # If no contour, use full dimensions
            total_area = width * height
            margin = CLOTH["EDGE_MARGIN"]
            usable_width = max(0, width - 2 * margin)
            usable_height = max(0, height - 2 * margin)
            usable_area = usable_width * usable_height
            return total_area, usable_area

        # Get pixel-to-cm scale
        img_height, img_width = cv2.boundingRect(contour)[2:4]
        scale_x = width / img_width if img_width > 0 else 1.0
        scale_y = height / img_height if img_height > 0 else 1.0

        # Calculate total area
        total_area = width * height

        # Calculate defect area
        defect_area = sum(
            cv2.contourArea(defect) * scale_x * scale_y for defect in defects
        )

        # Calculate area after edge margin
        margin = CLOTH["EDGE_MARGIN"]
        usable_width = max(0, width - 2 * margin)
        usable_height = max(0, height - 2 * margin)
        margin_area = usable_width * usable_height

        # Final usable area
        usable_area = margin_area - defect_area

        return total_area, max(0, usable_area)

    def detect_cloth_type(self, img: np.ndarray, contour: np.ndarray) -> str:
        """
        Detect cloth material type based on visual properties.
        Simple heuristic approach based on color and texture.
        """
        # Create mask for the cloth region
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Calculate average color in HSV space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]

        # Simple cloth type classification based on HSV values
        h, s, v = mean_hsv

        # Very simple rule-based classification using heuristics from config
        heuristics = CLOTH["TYPE_HEURISTICS"]
        if s < heuristics["saturation_threshold"]:  # Low saturation (grayscale)
            if v > heuristics["value_threshold"]:
                return "cotton"  # Light, neutral
            else:
                return "wool"  # Dark, neutral
        else:  # Has color
            if (
                h < heuristics["hue_red_lower"] or h > heuristics["hue_red_upper"]
            ):  # Reddish or purple
                return "silk"
            elif s > heuristics["saturation_vibrant"]:  # Vibrant colors
                return "polyester"
            else:
                return "mixed"

    def extract_material_properties(self, img: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Extract additional material properties like texture and grain direction.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Basic texture analysis
        properties = {}

        # Analyze for texture directionality (grain)
        try:
            # Create Gabor filters for different orientations
            orientations = []
            responses = []
            gabor_settings = CLOTH["GABOR_SETTINGS"]

            for theta in np.linspace(
                0, np.pi, gabor_settings["orientations"], endpoint=False
            ):
                kernel = cv2.getGaborKernel(
                    ksize=gabor_settings["ksize"],
                    sigma=gabor_settings["sigma"],
                    theta=theta,
                    lambd=gabor_settings["lambd"],
                    gamma=gabor_settings["gamma"],
                    psi=gabor_settings["psi"],
                )
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                mean_response = cv2.mean(filtered, mask=mask)[0]

                orientations.append(theta * 180 / np.pi)  # Convert to degrees
                responses.append(mean_response)

            # Get dominant direction
            max_idx = np.argmax(responses)
            grain_direction = orientations[max_idx]

            properties["grain_direction"] = grain_direction
            properties["texture_response"] = float(responses[max_idx])
        except Exception:
            # Fallback
            properties["grain_direction"] = 0
            properties["texture_response"] = 0

        return properties

    def process_image(self, image_path: str) -> ClothMaterial:
        """
        Process a cloth image and extract all relevant information.
        This is the main entry point for cloth recognition.
        """
        logger.info(f"Processing cloth image: {image_path}")

        # Extract cloth shape and defects
        shape_info = self.extract_cloth_shape(image_path)
        contour = shape_info["contour"]
        defects = shape_info["defects"]
        cloth_mask = shape_info["cloth_mask"]

        # Get filename and try to extract dimensions
        filename = os.path.basename(image_path)
        name = os.path.splitext(filename)[0]
        width, height = self.extract_dimensions_from_filename(filename)

        # If no dimensions in filename, use defaults
        if width is None or height is None:
            width = CLOTH["DEFAULT_WIDTH"]
            height = CLOTH["DEFAULT_HEIGHT"]
            logger.info(f"Using default dimensions: {width}x{height} cm")

        # Calculate areas
        total_area, usable_area = self.calculate_areas(contour, defects, width, height)

        # Detect material type and properties
        img = cv2.imread(image_path)
        cloth_type = self.detect_cloth_type(img, contour)

        # Extract additional properties if cloth mask is available
        material_properties = None
        if cloth_mask is not None:
            material_properties = self.extract_material_properties(img, cloth_mask)

        # Create ClothMaterial object
        cloth = ClothMaterial(
            id=hash(image_path),
            name=name,
            cloth_type=cloth_type,
            width=width,
            height=height,
            total_area=total_area,
            usable_area=usable_area,
            contour=contour,
            defects=defects,
            material_properties=material_properties,
        )

        logger.info(f"Cloth detected: {cloth_type}, {width:.1f}x{height:.1f} cm")
        logger.info(f"Usable area: {usable_area:.1f} cm² ({len(defects)} defects)")

        return cloth

    def visualize(self, cloth: ClothMaterial, output_path: str):
        """
        Create a visualization of cloth analysis for verification.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Draw cloth shape
        ax1.set_title(f"Cloth Shape: {cloth.cloth_type}")
        ax1.set_xlim(0, 512)
        ax1.set_ylim(0, 512)
        ax1.invert_yaxis()  # Match image coordinates
        ax1.set_aspect("equal")

        # Draw cloth contour
        if len(cloth.contour) > 0:
            contour = cloth.contour.squeeze()
            ax1.fill(
                contour[:, 0],
                contour[:, 1],
                color="lightblue",
                alpha=0.7,
                label="Cloth material",
            )
            ax1.plot(contour[:, 0], contour[:, 1], "b-", linewidth=2)

        # Draw defects
        for i, defect in enumerate(cloth.defects or []):
            defect_points = defect.squeeze()
            if len(defect_points.shape) > 1:
                ax1.fill(
                    defect_points[:, 0],
                    defect_points[:, 1],
                    color="red",
                    alpha=0.7,
                    label="Defect" if i == 0 else "",
                )

        ax1.legend()
        ax1.set_xlabel("Width (pixels)")
        ax1.set_ylabel("Height (pixels)")

        # Add cloth information
        ax2.axis("off")
        info_text = f"""Cloth Information:
Type: {cloth.cloth_type}
Dimensions: {cloth.width:.1f} x {cloth.height:.1f} cm
Total Area: {cloth.total_area:.1f} cm²
Usable Area: {cloth.usable_area:.1f} cm²
Utilization: {(cloth.usable_area / cloth.total_area * 100):.1f}%
Defects Found: {len(cloth.defects or [])}
Edge Margin: {CLOTH["EDGE_MARGIN"]} cm"""

        if cloth.material_properties:
            grain_dir = cloth.material_properties.get("grain_direction", "N/A")
            info_text += f"\nGrain Direction: {grain_dir:.1f}°"

        ax2.text(
            0.1,
            0.5,
            info_text,
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="center",
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Cloth analysis visualization saved to {output_path}")

    def save_model(self):
        """Save the cloth segmentation model."""
        if not self.use_unet or not hasattr(self, "model"):
            logger.warning("No model to save (U-Net not enabled)")
            return False

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "cloth_types": self.cloth_types,
            },
            self.model_path,
        )
        logger.info(f"Cloth segmentation model saved to {self.model_path}")
        return True

    def load_model(self) -> bool:
        """Load the cloth segmentation model."""
        if not self.use_unet or not hasattr(self, "model"):
            logger.warning("No model to load (U-Net not enabled)")
            return False

        if not os.path.exists(self.model_path):
            logger.warning(f"No model found at {self.model_path}")
            return False

        try:
            logger.info(f"Loading cloth segmentation model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Load cloth types if available
            if "cloth_types" in checkpoint:
                self.cloth_types = checkpoint["cloth_types"]

            logger.info("Cloth segmentation model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def train(
        self,
        train_images: List[str],
        val_images: List[str],
        epochs: int = 10,
        batch_size: int = 4,
    ) -> Dict:
        """Train the cloth segmentation model."""
        if not self.use_unet:
            logger.info("U-Net is not enabled, skipping training for cloth recognition.")
            return {}

        logger.info(
            f"Training cloth recognition model with {len(train_images)} train images and {len(val_images)} validation images"
        )

        # Transforms for training and validation
        transforms_set = transforms.Compose(
            [
                transforms.Resize((CLOTH["IMAGE_SIZE"], CLOTH["IMAGE_SIZE"])),
                transforms.ToTensor(),
            ]
        )

        # Create datasets and dataloaders
        train_dataset = ClothDataset(train_images, transforms=transforms_set)
        val_dataset = ClothDataset(val_images, transforms=transforms_set)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()

        return history
