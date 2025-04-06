import logging
from typing import Dict

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models

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
    """

    def __init__(
        self, num_cloth_types=10, encoder_name="resnet34", encoder_weights="imagenet"
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
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 2),  # Width and height
        )

        # Move models to device
        self.efficientnet = self.efficientnet.to(self.device)
        self.dim_mapper = self.dim_mapper.to(self.device)

        # Set models to evaluation mode
        self.efficientnet.eval()
        self.semantic_segmenter.eval()
        self.dim_mapper.eval()

    def process_cloth(self, image: np.ndarray) -> Dict:
        """Process cloth image and extract material properties"""
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
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 2. Apply Otsu thresholding based on image characteristics
            mean_value = np.mean(blurred)
            if mean_value > 127:
                # Dark cloth on light background
                _, initial_mask = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
            else:
                # Light cloth on dark background
                _, initial_mask = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

            # 3. Clean up the mask with morphological operations
            kernel_open = np.ones((5, 5), np.uint8)
            kernel_close = np.ones((7, 7), np.uint8)
            cleaned_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel_open)
            cleaned_mask = cv2.morphologyEx(
                cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2
            )

            # 4. Find contours in the cleaned mask
            contours, _ = cv2.findContours(
                cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 5. Find the largest contour (the cloth)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                min_area = 100

                if cv2.contourArea(main_contour) > min_area:
                    # Store the contour
                    results["contours"] = [main_contour]

                    # Create cloth mask
                    final_cloth_mask = np.zeros_like(gray)
                    cv2.drawContours(final_cloth_mask, [main_contour], -1, 255, -1)
                    results["cloth_mask"] = final_cloth_mask

                    # Calculate area
                    results["area"] = cv2.contourArea(main_contour)

                    # Get dimensions from bounding box
                    x, y, w, h = cv2.boundingRect(main_contour)
                    contour_dimensions = np.array([w, h], dtype=np.float32)

                    # Use contour dimensions if they seem valid
                    if w > 10 and h > 10:
                        logger.info("Using contour dimensions")
                        results["dimensions"] = contour_dimensions

                    # Create a polygon for boundary checking
                    try:
                        import shapely.geometry as sg

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

            # 6. Generate edge detection for visualization
            if np.any(results["cloth_mask"]):
                results["edges"] = cv2.Canny(results["cloth_mask"], 50, 150)
            else:
                results["edges"] = np.zeros_like(gray)

        except Exception as e:
            logger.error(f"Error during cloth processing: {e}")
            results["error"] = str(e)

            # Simple fallback approach
            try:
                gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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
                    results["edges"] = cv2.Canny(cloth_mask, 100, 200)
            except Exception as fallback_e:
                logger.error(f"Error during fallback processing: {fallback_e}")
                results["error"] = f"Main error: {e}; Fallback error: {fallback_e}"

        return results
