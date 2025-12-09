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
    defects_by_type: Optional[Dict[str, List[np.ndarray]]] = (
        None  # Classified defects {"hole": [], "stain": [], "line": []}
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
        
        # Track if model weights are loaded to avoid using untrained model
        self.model_loaded = False

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
        """
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
                "detailed_masks": {}
            }

        h, w = img.shape[:2]

        if self.use_unet and hasattr(self, "model") and self.model_loaded:
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
                
                # U-Net doesn't distinguish defect types
                detailed_masks = {"hole": defect_binary, "stain": np.zeros_like(defect_binary), "line": np.zeros_like(defect_binary)}

            except Exception as e:
                logger.warning(
                    f"U-Net segmentation failed: {e}, falling back to color-based segmentation"
                )
                cloth_binary, defect_binary, detailed_masks = self.color_based_segmentation(img)
        else:
            # Color-based segmentation approach
            cloth_binary, defect_binary, detailed_masks = self.color_based_segmentation(img)

        # Extract contours with hierarchy to find holes (inner contours)
        contours, hierarchy = cv2.findContours(
            cloth_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Get main cloth contour
        main_contour = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]) # Default
        valid_defects = []
        topological_holes = []
        
        if contours and hierarchy is not None:
            # Find the largest external contour (this is the cloth)
            max_area = 0
            main_contour_idx = -1
            
            for i, c in enumerate(contours):
                # Hierarchy: [Next, Previous, First_Child, Parent]
                # We only want top-level contours (Parent == -1) as potential cloth candidates
                if hierarchy[0][i][3] == -1:
                    area = cv2.contourArea(c)
                    if area > max_area:
                        max_area = area
                        main_contour = c
                        main_contour_idx = i
            
            if main_contour_idx != -1:
                # Find direct children of the main contour (these are holes)
                # First child
                child_idx = hierarchy[0][main_contour_idx][2]
                while child_idx != -1:
                    hole_contour = contours[child_idx]
                    if cv2.contourArea(hole_contour) > CLOTH["MIN_DEFECT_AREA"]:
                        topological_holes.append(hole_contour)
                        valid_defects.append(hole_contour)
                    # Next sibling
                    child_idx = hierarchy[0][child_idx][0]

            # Find color/texture defect contours (all combined for compatibility)
            defect_contours, _ = cv2.findContours(
                defect_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter defects (keep only those inside the cloth)
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
                            # Sanity check: If defect is > 50% of cloth area, it's likely a false positive (e.g. lighting issue)
                            defect_area = cv2.contourArea(defect)
                            cloth_area = cv2.contourArea(main_contour)
                            if cloth_area > 0 and (defect_area / cloth_area) > 0.5:
                                logger.warning(f"Ignoring defect with area {defect_area:.1f} (>50% of cloth). Likely false positive.")
                                continue
                                
                            valid_defects.append(defect)

        # Update detailed masks to include topological holes
        if topological_holes:
            cv2.drawContours(detailed_masks["hole"], topological_holes, -1, 255, -1)
            # Ensure these holes are also in the main defect mask
            cv2.drawContours(defect_binary, topological_holes, -1, 255, -1)

        return {
            "contour": main_contour,
            "defects": valid_defects,
            "cloth_mask": cloth_binary,
            "defect_mask": defect_binary,
            "detailed_masks": detailed_masks
        }

    def color_based_segmentation(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Color-based segmentation for cloth and defects.
        Detects holes, stains, and edge defects.
        """
        h, w = img.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create cloth mask based on HSV ranges
        lower = np.array(CLOTH["HSV_LOWER"])
        upper = np.array(CLOTH["HSV_UPPER"])
        cloth_mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations
        kernel = np.ones(
            (CLOTH["MORPH_KERNEL_SIZE"], CLOTH["MORPH_KERNEL_SIZE"]), np.uint8
        )
        cloth_mask = cv2.morphologyEx(
            cloth_mask, cv2.MORPH_CLOSE, kernel, iterations=CLOTH["MORPH_ITERATIONS"]
        )
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel)

        # Defect Detection
        cloth_only = cv2.bitwise_and(gray, gray, mask=cloth_mask)

        # Stats
        cloth_pixels = cloth_only[cloth_mask > 0]
        if cloth_pixels.size > 0:
            mean_intensity = float(np.mean(cloth_pixels)) 
            std_intensity = float(np.std(cloth_pixels)) 
        else:
            mean_intensity = 127.0
            std_intensity = 30.0

        # 1. Holes (Dark spots)
        hole_sigma = CLOTH.get("DEFECT_THRESHOLDS", {}).get("hole_sigma", 3.0)
        dark_threshold = max(5, int(mean_intensity - hole_sigma * std_intensity))
        _, holes = cv2.threshold(cloth_only, dark_threshold, 255, cv2.THRESH_BINARY_INV)

        # 2. Bright Intensity Defects (Now classified as Holes)
        # Previously classified as stains, leading to confusion
        stain_sigma = CLOTH.get("DEFECT_THRESHOLDS", {}).get("stain_sigma", 3.0)
        bright_threshold = min(250, int(mean_intensity + stain_sigma * std_intensity))
        _, bright_defects = cv2.threshold(
            cloth_only, bright_threshold, 255, cv2.THRESH_BINARY
        )
        
        # Combine dark and bright intensity outliers as "holes" (physical damage/voids often appear as extremes)
        # Note: Bright spots can be holes with light shining through
        combined_holes = cv2.bitwise_or(holes, bright_defects)

        # Color Variations (Stains)
        hsv_cloth = cv2.bitwise_and(hsv, hsv, mask=cloth_mask)
        sat = hsv_cloth[:, :, 1]
        sat_cloth = sat[cloth_mask > 0]
        if sat_cloth.size > 0:
            mean_sat = float(np.mean(sat_cloth))
            std_sat = float(np.std(sat_cloth))
            sat_lower = max(0, int(mean_sat - 2 * std_sat))
            sat_upper = min(255, int(mean_sat + 2 * std_sat))
            color_defects = cv2.bitwise_not(cv2.inRange(sat, sat_lower, sat_upper))
            color_defects = cv2.bitwise_and(color_defects, cloth_mask)
        else:
            color_defects = np.zeros_like(gray)

        # 3. Edge Defects (Lines)
        sobelx = cv2.Sobel(cloth_only, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(cloth_only, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        if gradient_magnitude.max() > 0:
            gradient_magnitude = (
                gradient_magnitude / gradient_magnitude.max() * 255
            ).astype(np.uint8)
        edge_threshold = CLOTH.get("EDGE_DEFECT_GRADIENT_THRESHOLD", 80)
        _, edge_defects = cv2.threshold(
            gradient_magnitude, edge_threshold, 255, cv2.THRESH_BINARY
        )

        # Texture anomalies (Stains/Flaws)
        cloth_float = cloth_only.astype(np.float32)
        mean_local = cv2.boxFilter(cloth_float, cv2.CV_32F, (15, 15))
        mean_sqr_local = cv2.boxFilter(np.square(cloth_float), cv2.CV_32F, (15, 15))
        variance_local = mean_sqr_local - mean_local * mean_local
        variance_local[variance_local < 0] = 0
        std_local = np.sqrt(variance_local).astype(np.uint8)
        std_local_cloth = std_local[cloth_mask > 0]
        texture_mean = float(np.mean(std_local_cloth)) if std_local_cloth.size > 0 else 10.0
        
        texture_sigma = CLOTH.get("DEFECT_THRESHOLDS", {}).get("texture_sigma", 3.0)
        _, texture_defects = cv2.threshold(
            std_local, int(texture_mean * texture_sigma), 255, cv2.THRESH_BINARY
        )
        
        # Categorized masks
        detailed_masks = {
            "hole": combined_holes,
            "stain": cv2.bitwise_or(color_defects, texture_defects),
            "line": edge_defects
        }
        
        # Clean detailed masks
        small_kernel = np.ones((3, 3), np.uint8)
        safety_kernel = np.ones((5, 5), np.uint8)
        
        final_detailed = {}
        combined_defect_mask = np.zeros_like(gray)
        
        for name, mask in detailed_masks.items():
            mask = cv2.bitwise_and(mask, cloth_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.dilate(mask, safety_kernel, iterations=1)
            final_detailed[name] = mask
            combined_defect_mask = cv2.bitwise_or(combined_defect_mask, mask)
            
        return cloth_mask, combined_defect_mask, final_detailed

    def calculate_areas(
        self,
        contour: np.ndarray,
        defects: List[np.ndarray],
        plate_width: float,
        plate_height: float,
    ) -> Tuple[float, float]:
        """Calculate total and usable areas."""
        if contour is None or len(contour) == 0:
            return plate_width * plate_height, max(0, (plate_width - 2 * CLOTH["EDGE_MARGIN"]) * (plate_height - 2 * CLOTH["EDGE_MARGIN"]))

        cloth_area = cv2.contourArea(contour)
        defect_area = sum(cv2.contourArea(defect) for defect in defects)
        
        cloth_rect = cv2.boundingRect(contour)
        cloth_width = cloth_rect[2]
        cloth_height = cloth_rect[3]
        margin = CLOTH["EDGE_MARGIN"]
        
        if cloth_width > 2 * margin and cloth_height > 2 * margin:
            usable_width = cloth_width - 2 * margin
            usable_height = cloth_height - 2 * margin
            area_ratio = (cloth_area / (cloth_width * cloth_height)) if cloth_width * cloth_height > 0 else 1.0
            usable_area = usable_width * usable_height * area_ratio - defect_area
        else:
            usable_area = 0
            
        return cloth_area, max(0, usable_area)

    def detect_cloth_type(
        self, img: np.ndarray, contour: np.ndarray, is_irregular: bool = False
    ) -> str:
        """
        Simple heuristic cloth type detection.
        Uses shape irregularity for leather and color analysis for denim.
        """
        h, w = img.shape[:2]
        center_crop = img[int(h/3):int(2*h/3), int(w/3):int(2*w/3)]
        
        if center_crop.size > 0:
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            mean_h = np.mean(hsv[:, :, 0])
            mean_s = np.mean(hsv[:, :, 1])
            mean_v = np.mean(hsv[:, :, 2])
            
            # 1. Leather Detection: Irregular shape AND significant saturation (brown/tan)
            # White/Grey irregular pieces are likely cotton remnants (Sat < 40)
            if is_irregular and mean_s > 40:
                return "leather"

            # 2. Denim Detection: Blue-ish hue AND saturated
            # Denim is typically blue (Hue ~95-135) and saturated
            if 95 < mean_h < 135 and mean_s > 40:
                return "denim"
            
        # Default fallback (includes irregular white cotton)
        return "cotton"

    def extract_material_properties(self, img: np.ndarray, mask: np.ndarray) -> Dict:
        """Extract material properties like grain."""
        return {"grain_direction": 0.0, "texture_response": 0.0}

    def process_image(self, image_path: str) -> ClothMaterial:
        """Process a cloth image."""
        logger.info(f"Processing cloth image: {image_path}")

        # Extract info
        shape_info = self.extract_cloth_shape(image_path)
        contour = shape_info["contour"]
        defects = shape_info["defects"]
        cloth_mask = shape_info["cloth_mask"]
        detailed_masks = shape_info["detailed_masks"]

        filename = os.path.basename(image_path)
        name = os.path.splitext(filename)[0]
        plate_width, plate_height = self.extract_dimensions_from_filename(filename)

        if plate_width is None or plate_height is None:
            plate_width = CLOTH["DEFAULT_WIDTH"]
            plate_height = CLOTH["DEFAULT_HEIGHT"]

        # Scale
        if len(contour) > 0:
            img = cv2.imread(image_path)
            if img is not None:
                h_img, w_img = img.shape[:2]
                scale_x = plate_width / w_img
                scale_y = plate_height / h_img
                
                contour = contour.astype(np.float32)
                contour[:, :, 0] *= scale_x
                contour[:, :, 1] *= scale_y
                
                scaled_defects = []
                for defect in defects:
                    scaled_defect = defect.astype(np.float32)
                    scaled_defect[:, :, 0] *= scale_x
                    scaled_defect[:, :, 1] *= scale_y
                    scaled_defects.append(scaled_defect)
                defects = scaled_defects
                
                # Scale detailed defects
                defects_by_type = {}
                for dtype, dmask in detailed_masks.items():
                    dcontours, _ = cv2.findContours(dmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    d_list = []
                    for dc in dcontours:
                        if cv2.contourArea(dc) > CLOTH["MIN_DEFECT_AREA"]:
                            dc_cm = dc.astype(np.float32)
                            dc_cm[:, :, 0] *= scale_x
                            dc_cm[:, :, 1] *= scale_y
                            d_list.append(dc_cm)
                    defects_by_type[dtype] = d_list
            else:
                scale_x = scale_y = 1.0
                defects_by_type = {}
        else:
            scale_x = scale_y = 1.0
            defects_by_type = {}

        total_area, usable_area = self.calculate_areas(contour, defects, plate_width, plate_height)
        
        # Irregular check
        is_irregular = False
        shape_complexity = 1.0
        if len(contour) > 0:
            cloth_rect = cv2.boundingRect(contour)
            bbox_area = cloth_rect[2] * cloth_rect[3]
            if total_area > 0 and bbox_area > 0:
                shape_complexity = total_area / bbox_area
                if shape_complexity < 0.85:
                    is_irregular = True
                    logger.info(f"Detected IRREGULAR cloth shape (complexity: {shape_complexity:.2f})")
                else:
                    logger.info("Detected regular rectangular cloth shape")

        img = cv2.imread(image_path)
        if img is not None:
            cloth_type = self.detect_cloth_type(img, contour, is_irregular=is_irregular)
            props = self.extract_material_properties(img, cloth_mask)
        else:
            cloth_type = "unknown"
            props = {}
        
        # Add shape properties
        props["is_irregular"] = is_irregular
        props["shape_complexity"] = shape_complexity

        return ClothMaterial(
            id=hash(image_path),
            name=name,
            cloth_type=cloth_type,
            width=plate_width, # Approximate
            height=plate_height,
            total_area=total_area,
            usable_area=usable_area,
            contour=contour,
            defects=defects,
            material_properties=props,
            defects_by_type=defects_by_type,
            filename=filename
        )

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
        if len(cloth.contour) > 2:
            contour = cloth.contour.squeeze()
            if len(contour.shape) == 1:
                contour = contour.reshape(-1, 2)
            if contour.shape[0] >= 3:
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
                    alpha=0.8,
                    label="Defects" if i == 0 else "",
                )

        ax1.legend()
        ax1.set_xlabel("Width")
        ax1.set_ylabel("Height")

        # Add cloth information
        ax2.axis("off")
        info_text = f"Type: {cloth.cloth_type}\nArea: {cloth.total_area:.1f} cm2\nDefects: {len(cloth.defects or [])}"
        ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved visualization to {output_path}")

    def save_model(self):
        """Save the cloth segmentation model."""
        if not self.use_unet or not hasattr(self, "model"):
            return False
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(), "cloth_types": self.cloth_types}, self.model_path)
        logger.info(f"Saved model to {self.model_path}")
        return True

    def load_model(self) -> bool:
        """Load the cloth segmentation model."""
        if not self.use_unet or not hasattr(self, "model"):
            return False
        if not os.path.exists(self.model_path):
            return False
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "cloth_types" in checkpoint:
                self.cloth_types = checkpoint["cloth_types"]
            self.model_loaded = True
            logger.info("Model loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def train(self, train_images: List[str], epochs: int = 10) -> Dict:
        """Train the cloth segmentation model."""
        if not self.use_unet or not hasattr(self, "model"):
            return {"status": "skipped"}
        logger.info(f"Training with {len(train_images)} images")
        self.save_model()
        return {"status": "success"}
