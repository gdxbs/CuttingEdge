import logging
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
import shapely.geometry as sg

from cutting_edge.config import (
    IMAGE_PROCESSING,
    MODEL,
    TRAINING,
    VISUALIZATION,
)
from cutting_edge.utils import preprocess_image_for_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClothRecognitionModule:
    """Module for cloth material recognition and dimension mapping

    This module analyzes cloth materials in images to identify:
    - Cloth type classification
    - Semantic segmentation to separate cloth from background
    - Dimension estimation of the cloth
    - Boundary information for pattern fitting
    """

    def __init__(
        self, num_cloth_types=MODEL["DEFAULT_CLOTH_TYPES"], 
        encoder_name=MODEL["DEFAULT_ENCODER"], 
        encoder_weights="imagenet"
    ):
        """Initialize the cloth recognition module"""
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained EfficientNet for classification
        self.efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Adjust the classifier to our number of cloth types
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_cloth_types)

        # Load U-Net for semantic segmentation
        self.semantic_segmenter = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,  # RGB images
            classes=num_cloth_types,
        ).to(self.device)

        # Create a dimension prediction network
        self.dim_mapper = nn.Sequential(
            nn.Linear(num_ftrs, MODEL["FEATURE_DIM"]),
            nn.ReLU(),
            nn.Linear(MODEL["FEATURE_DIM"], 2),  # Width and height
        )

        # Move models to device
        self.efficientnet = self.efficientnet.to(self.device)
        self.dim_mapper = self.dim_mapper.to(self.device)

        # Set models to evaluation mode
        self.efficientnet.eval()
        self.semantic_segmenter.eval()
        self.dim_mapper.eval()

    def process_cloth(self, image: np.ndarray) -> Dict:
        """Process cloth image and extract the largest area for pattern fitting"""
        if image is None:
            raise ValueError("Input cloth image cannot be None")

        # Make a copy to avoid modifying the original
        image_copy = image.copy()

        # Ensure we have a 3-channel image
        if len(image_copy.shape) == 2:  # Convert grayscale to BGR
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)

        h_orig, w_orig = image_copy.shape[:2]

        # Default return values
        results = {
            "features": None,
            "contours": [],
            "dimensions": np.array([w_orig, h_orig], dtype=np.float32),
            "edges": None,
            "segmented_image": None,
            "area": 0.0,
            "cloth_mask": np.zeros((h_orig, w_orig), dtype=np.uint8),
            "error": None,
            "cloth_polygon": None,  # Will contain the shapely polygon for boundary checks
            "visualization_mask": None,  # Resized mask for visualization
        }

        try:
            # Preprocess image for deep learning models
            processed_image = preprocess_image_for_model(image_copy, self.device)

            with torch.no_grad():
                # Extract features from EfficientNet
                x = self.efficientnet.features(processed_image)
                x = self.efficientnet.avgpool(x)
                features = torch.flatten(x, 1)

                # Predict dimensions
                predicted_dimensions = self.dim_mapper(features).cpu().numpy()[0]
                results["features"] = features.cpu().numpy()
                results["dimensions"] = predicted_dimensions

                # Run semantic segmentation
                logits = self.semantic_segmenter(processed_image)
                mask = torch.argmax(logits, dim=1, keepdim=True)

                # Resize back to original dimensions
                if mask is not None:
                    segmented_map = cv2.resize(
                        mask.cpu().numpy().squeeze(),
                        (w_orig, h_orig),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    results["segmented_image"] = segmented_map

            # Traditional image processing for cloth detection
            # 1. Convert to grayscale and blur
            gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, IMAGE_PROCESSING["MORPH_KERNEL_SIZE"], 0)

            # 2. Apply Otsu thresholding for more reliable cloth extraction
            _, initial_mask = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # 3. Clean up the mask with morphological operations
            kernel_open = np.ones((5, 5), np.uint8)
            kernel_close = np.ones((15, 15), np.uint8)
            
            # First dilate to make sure the cloth is fully captured
            dilated_mask = cv2.dilate(initial_mask, kernel_open, iterations=2)
            
            # Then close gaps and holes
            cleaned_mask = cv2.morphologyEx(
                dilated_mask, cv2.MORPH_CLOSE, kernel_close, 
                iterations=IMAGE_PROCESSING["MORPH_CLOSE_ITERATIONS"]
            )

            # 4. Find contours in the cleaned mask
            contours, _ = cv2.findContours(
                cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 5. Filter contours based on criteria
            filtered_contours = self._filter_contours(contours)

            # 6. Find the largest contour (the cloth)
            if filtered_contours:
                main_contour = max(filtered_contours, key=cv2.contourArea)
                
                # Store the contour
                results["contours"] = [main_contour]

                # Create cloth mask
                final_cloth_mask = np.zeros_like(gray)
                cv2.drawContours(final_cloth_mask, [main_contour], -1, 255, -1)
                
                # Fill any small holes in the mask
                kernel = np.ones((5, 5), np.uint8)
                final_cloth_mask = cv2.morphologyEx(final_cloth_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                # Store the mask
                results["cloth_mask"] = final_cloth_mask

                # Calculate area
                results["area"] = cv2.contourArea(main_contour)

                # Get dimensions from bounding box
                x, y, w, h = cv2.boundingRect(main_contour)
                contour_dimensions = np.array([w, h], dtype=np.float32)

                # Use contour dimensions if they seem valid
                if w > IMAGE_PROCESSING["MIN_DIMENSION"] and h > IMAGE_PROCESSING["MIN_DIMENSION"]:
                    logger.info("Using contour dimensions")
                    results["dimensions"] = contour_dimensions

                # Create a polygon for boundary checking
                try:
                    # Convert contour to polygon format for shapely
                    contour_points = main_contour.squeeze()
                    # Ensure contour is closed
                    if not np.array_equal(contour_points[0], contour_points[-1]):
                        contour_points = np.vstack(
                            [contour_points, contour_points[0]]
                        )

                    # Create polygon
                    results["cloth_polygon"] = sg.Polygon(contour_points)

                    # Pre-generate visualization mask for all potential sizes
                    results["visualization_mask"] = final_cloth_mask
                except Exception as poly_err:
                    logger.warning(f"Failed to create cloth polygon: {poly_err}")

            # 7. Generate edge detection for visualization
            if np.any(results["cloth_mask"]):
                # Use dilated edges for better visibility
                edges = cv2.Canny(
                    results["cloth_mask"], 
                    IMAGE_PROCESSING["EDGE_DETECTION_LOW"], 
                    IMAGE_PROCESSING["EDGE_DETECTION_HIGH"]
                )
                # Dilate edges to make them more visible
                edge_kernel = np.ones((3, 3), np.uint8)
                results["edges"] = cv2.dilate(edges, edge_kernel, iterations=1)
            else:
                results["edges"] = np.zeros_like(gray)

        except Exception as e:
            logger.error(f"Error during cloth processing: {e}")
            results["error"] = str(e)

            # Simple fallback approach
            try:
                gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, IMAGE_PROCESSING["MORPH_KERNEL_SIZE"], 0)
                _, initial_mask = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                contours, _ = cv2.findContours(
                    initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    results["dimensions"] = np.array([w, h], dtype=np.float32)
                    results["contours"] = [largest_contour]
                    results["area"] = cv2.contourArea(largest_contour)

                    # Create mask from contour
                    cloth_mask = np.zeros_like(gray)
                    cv2.drawContours(cloth_mask, [largest_contour], -1, 255, -1)
                    results["cloth_mask"] = cloth_mask
                    results["edges"] = cv2.Canny(
                        cloth_mask, 
                        IMAGE_PROCESSING["EDGE_DETECTION_LOW"], 
                        IMAGE_PROCESSING["EDGE_DETECTION_HIGH"]
                    )
            except Exception as fallback_e:
                logger.error(f"Error during fallback processing: {fallback_e}")
                results["error"] = f"Main error: {e}; Fallback error: {fallback_e}"
        
        return results

    def _filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Filter contours based on area, perimeter, and shape criteria"""
        filtered_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Skip contours that are too small
            if area < IMAGE_PROCESSING["CONTOUR_MIN_AREA"] or perimeter < IMAGE_PROCESSING["CONTOUR_MIN_PERIMETER"]:
                continue
                
            # Approximate the contour to reduce noise and simplify shape
            epsilon = IMAGE_PROCESSING["CONTOUR_APPROX_EPSILON"] * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Skip if approximation has too many points (likely noise)
            if len(approx) > IMAGE_PROCESSING["CONTOUR_MAX_APPROX_POINTS"]:
                continue
                
            filtered_contours.append(contour)
            
        return filtered_contours
