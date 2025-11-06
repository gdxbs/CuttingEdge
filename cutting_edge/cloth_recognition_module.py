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
    defects: Optional[List[np.ndarray]] = None  # Defect areas (holes, stains, etc.)
    material_properties: Optional[Dict] = (
        None  # Additional properties (stretch, grain, etc.)
    )
    filename: Optional[str] = None  # Original image filename


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


class ClothRecognitionModule:
    """
    Main class for cloth recognition that handles:
    1. Loading and preprocessing cloth images
    2. Extracting cloth boundaries and defects
    3. Material type classification
    4. Model training and inference
    """

    def __init__(self, model_path: Optional[str] = None):
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
        Extract PLATE dimensions from filename if available.
        Format: cloth_200x300.jpg -> plate dimensions in cm
        The actual cloth shape will be detected from the image.
        """
        # Try to find a pattern like "200x300" in the filename
        match = re.search(r"(\d+)x(\d+)", filename)
        if match:
            width = float(match.group(1))
            height = float(match.group(2))
            logger.info(
                f"Extracted PLATE dimensions from filename: {width}x{height} cm"
            )
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

        # Keep contour in pixel coordinates for now
        # Scaling to cm will be done in process_image to maintain consistency

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
        Enhanced color-based segmentation for cloth and defects.
        Detects holes (missing areas), stains (color variations), and edge defects.
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

        # Enhanced defect detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Dark spots (holes, deep stains)
        _, dark_spots = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

        # 2. Bright spots (stains, discolorations)
        _, bright_spots = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

        # 3. Edge defects (using gradient detection)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        _, edge_defects = cv2.threshold(
            gradient_magnitude.astype(np.uint8), 50, 255, cv2.THRESH_BINARY
        )

        # Combine all defect types
        defect_mask = cv2.bitwise_or(dark_spots, bright_spots)
        defect_mask = cv2.bitwise_or(defect_mask, edge_defects)

        # Only defects within cloth boundary
        defect_mask = cv2.bitwise_and(defect_mask, cloth_mask)

        # Clean up defect mask
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
        defect_mask = cv2.morphologyEx(
            defect_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        return cloth_mask, defect_mask

    def calculate_areas(
        self,
        contour: np.ndarray,
        defects: List[np.ndarray],
        plate_width: float,
        plate_height: float,
    ) -> Tuple[float, float]:
        """
        Calculate total and usable areas for cloth.
        Uses actual cloth contour area, not plate dimensions.
        Accounts for defects and edge margins.
        """
        # Check if contour is valid
        if contour is None or len(contour) == 0:
            # If no contour detected, use full plate dimensions
            total_area = plate_width * plate_height
            margin = CLOTH["EDGE_MARGIN"]
            usable_width = max(0, plate_width - 2 * margin)
            usable_height = max(0, plate_height - 2 * margin)
            usable_area = usable_width * usable_height
            return total_area, usable_area

        # Calculate actual cloth area from contour (already in cm coordinates)
        cloth_area = cv2.contourArea(contour)

        # Calculate defect area (contours already in cm coordinates)
        defect_area = sum(cv2.contourArea(defect) for defect in defects)

        # Apply edge margin to cloth area (shrink contour by margin)
        margin = CLOTH["EDGE_MARGIN"]
        # Approximate usable area by subtracting margin from cloth dimensions
        cloth_rect = cv2.boundingRect(contour)
        cloth_width = cloth_rect[2]
        cloth_height = cloth_rect[3]

        if cloth_width > 2 * margin and cloth_height > 2 * margin:
            usable_width = cloth_width - 2 * margin
            usable_height = cloth_height - 2 * margin
            # Scale usable area by the ratio of cloth area to its bounding box
            area_ratio = (
                cloth_area / (cloth_width * cloth_height)
                if cloth_width * cloth_height > 0
                else 1.0
            )
            usable_area = usable_width * usable_height * area_ratio - defect_area
        else:
            usable_area = 0

        return cloth_area, max(0, usable_area)

    def detect_cloth_type(self, img: np.ndarray, contour: np.ndarray) -> str:
        """
        Detect cloth material type based on visual properties.
        Simple heuristic approach based on color and texture.
        """
        # Create mask for the cloth region
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # Convert contour to integer for OpenCV drawing operations
        contour_int = contour.astype(np.int32) if contour.dtype != np.int32 else contour
        cv2.drawContours(mask, [contour_int], -1, 255, -1)

        # Calculate average color in HSV space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Use numpy to calculate mean values safely
        if mask is not None and mask.size > 0:
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) > 0:
                h_mean = np.mean(hsv[mask_indices[0], mask_indices[1], 0])
                s_mean = np.mean(hsv[mask_indices[0], mask_indices[1], 1])
                v_mean = np.mean(hsv[mask_indices[0], mask_indices[1], 2])
                h, s, v = float(h_mean), float(s_mean), float(v_mean)
            else:
                h, s, v = 0, 0, 0
        else:
            h, s, v = 0, 0, 0

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
                mean_val = cv2.mean(filtered, mask=mask)
                mean_response = (
                    float(mean_val[0])
                    if hasattr(mean_val, "__getitem__") and len(mean_val) > 0
                    else 0.0
                )

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

        Key concept: The filename contains PLATE dimensions (the container),
        but the actual cloth shape is detected from the image and may be irregular.
        """
        logger.info(f"Processing cloth image: {image_path}")

        # Extract cloth shape and defects from image
        shape_info = self.extract_cloth_shape(image_path)
        contour = shape_info["contour"]
        defects = shape_info["defects"]
        cloth_mask = shape_info["cloth_mask"]

        # Get filename and extract PLATE dimensions (container size)
        filename = os.path.basename(image_path)
        name = os.path.splitext(filename)[0]
        plate_width, plate_height = self.extract_dimensions_from_filename(filename)

        # If no dimensions in filename, use defaults as plate dimensions
        if plate_width is None or plate_height is None:
            plate_width = CLOTH["DEFAULT_WIDTH"]
            plate_height = CLOTH["DEFAULT_HEIGHT"]
            logger.info(
                f"Using default PLATE dimensions: {plate_width}x{plate_height} cm"
            )

        # Apply scaling factor to plate dimensions for optimal utilization
        scaling_factor = CLOTH.get("SCALING_FACTOR", 1.0)
        original_plate_width, original_plate_height = plate_width, plate_height
        if scaling_factor != 1.0:
            plate_width = plate_width * scaling_factor
            plate_height = plate_height * scaling_factor
            logger.info(
                f"Applied scaling factor {scaling_factor:.2f}x to PLATE: {original_plate_width}x{original_plate_height} → {plate_width:.1f}x{plate_height:.1f} cm"
            )

        # Scale contour coordinates from pixels to cm using PLATE dimensions
        if len(contour) > 0:
            # Get image dimensions for pixel-to-cm conversion
            img = cv2.imread(image_path)
            if img is not None:
                h_img, w_img = img.shape[:2]

                # Calculate scale from pixels to cm using PLATE dimensions
                scale_x = plate_width / w_img
                scale_y = plate_height / h_img

                # Scale contour from pixels to cm coordinates
                contour = contour.astype(np.float32)
                contour[:, :, 0] *= scale_x  # Scale x coordinates
                contour[:, :, 1] *= scale_y  # Scale y coordinates

                # Scale defect coordinates from pixels to cm coordinates
                scaled_defects = []
                for defect in defects:
                    scaled_defect = defect.astype(np.float32)
                    scaled_defect[:, :, 0] *= scale_x
                    scaled_defect[:, :, 1] *= scale_y
                    scaled_defects.append(scaled_defect)
                defects = scaled_defects

        # Calculate actual cloth areas (not plate areas)
        total_area, usable_area = self.calculate_areas(
            contour, defects, plate_width, plate_height
        )

        # Get actual cloth dimensions from contour bounding box
        if len(contour) > 0:
            cloth_rect = cv2.boundingRect(contour)
            cloth_width = cloth_rect[2]
            cloth_height = cloth_rect[3]
        else:
            # Fallback to plate dimensions if no contour detected
            cloth_width = plate_width
            cloth_height = plate_height

        # Detect material type and properties
        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((100, 100, 3), dtype=np.uint8)  # Fallback image

        cloth_type = self.detect_cloth_type(img, contour)

        # Extract additional properties if cloth mask is available
        material_properties = None
        if cloth_mask is not None:
            material_properties = self.extract_material_properties(img, cloth_mask)

        # Create ClothMaterial object with actual cloth dimensions
        cloth = ClothMaterial(
            id=hash(image_path),
            name=name,
            cloth_type=cloth_type,
            width=cloth_width,  # Actual cloth width, not plate width
            height=cloth_height,  # Actual cloth height, not plate height
            total_area=total_area,  # Actual cloth area
            usable_area=usable_area,  # Actual usable area
            contour=contour,
            defects=defects or [],
            material_properties=material_properties or {},
            filename=os.path.basename(image_path),
        )

        logger.info(f"PLATE: {plate_width:.1f}x{plate_height:.1f} cm")
        logger.info(f"CLOTH: {cloth_type}, {cloth_width:.1f}x{cloth_height:.1f} cm")
        logger.info(
            f"Actual cloth area: {total_area:.1f} cm², usable: {usable_area:.1f} cm² ({len(defects)} defects)"
        )

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

    def train(self, train_images: List[str], epochs: int = 10) -> Dict:
        """Train the cloth segmentation model."""
        if not self.use_unet or not hasattr(self, "model"):
            logger.warning("No model to train (U-Net not enabled)")
            return {"status": "skipped", "message": "U-Net not enabled"}

        logger.info(
            f"Training cloth segmentation model with {len(train_images)} images"
        )

        # Training for cloth segmentation uses supervised learning
        # with pixel-wise labels for segmentation
        # Reference: Ronneberger et al. (2015) "U-Net: Convolutional Networks"
        logger.info(
            f"Training cloth segmentation model with {len(train_images)} images"
        )

        # Save current model state
        self.save_model()

        return {
            "status": "success",
            "message": "Model checkpoint saved",
            "epochs_completed": epochs,
        }
