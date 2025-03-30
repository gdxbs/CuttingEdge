from typing import Dict

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class ClothRecognitionModule:
    """Module for cloth material recognition and dimension mapping

    This module analyzes cloth materials in images to identify properties such as:
    - Cloth type classification (e.g., cotton, silk, denim)
    - Semantic segmentation of cloth regions
    - Dimension estimation

    The architecture uses multiple specialized networks:
    1. EfficientNet-B0 for cloth type classification
    2. U-Net with ResNet34 backbone for semantic segmentation
    3. Custom MLP for dimension mapping

    References:
    - EfficientNet: "EfficientNet: Rethinking Model Scaling for CNNs" (Tan & Le, 2019)
      https://arxiv.org/abs/1905.11946
    - U-Net: "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)
      https://arxiv.org/abs/1505.04597
    - Semantic segmentation adapted from: segmentation_models.pytorch library
      https://github.com/qubvel/segmentation_models.pytorch
    """

    def __init__(
        self, num_cloth_types=10, encoder_name="resnet34", encoder_weights="imagenet"
    ):
        """Initialize the cloth recognition module

        Args:
            num_cloth_types: Number of cloth material types to classify
            encoder_name: Name of encoder backbone for U-Net (default: "resnet34")
            encoder_weights: Pretrained weights for encoder (default: "imagenet")
        """
        # Set computation device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cloth type classification model using EfficientNet-B0
        # EfficientNet achieves better accuracy with fewer parameters than other CNNs
        # REF: "EfficientNet: Rethinking Model Scaling for CNNs" (Tan & Le, 2019)
        # https://arxiv.org/abs/1905.11946
        self.efficientnet = models.efficientnet_b0(
            pretrained=True
        )  # Pretrained on ImageNet-1k

        # Modify the classifier head to output the correct number of cloth types
        # The original outputs 1000 classes (ImageNet), we change it to num_cloth_types
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, num_cloth_types
        )

        # Semantic segmentation model using U-Net architecture
        # U-Net is effective for precise segmentation tasks due to its
        # encoder-decoder structure with skip connections
        # REF: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        # https://arxiv.org/abs/1505.04597
        self.semantic_segmenter = smp.Unet(
            encoder_name=encoder_name,  # ResNet34 provides good accuracy/speed balance
            encoder_weights=encoder_weights,  # Using ImageNet pretrained weights
            in_channels=3,  # RGB input images
            classes=num_cloth_types,  # Output channels match cloth types
        ).to(self.device)

        # Dimension mapping network (MLP) for estimating cloth dimensions
        # EfficientNet-B0 feature extractor outputs 1280-dim features
        # This is the standard size for EfficientNet-B0's features after pooling
        efficientnet_features = 1280  # Fixed size for EfficientNet-B0

        # Takes EfficientNet features and predicts width/height (2-dim)
        self.dim_mapper = nn.Sequential(
            nn.Linear(efficientnet_features, 512),  # First layer reduces features
            nn.ReLU(),  # ReLU activation for non-linearity
            nn.Linear(512, 2),  # Final layer outputs width and height
        )

        # Move models to the computation device (GPU/CPU)
        self.efficientnet = self.efficientnet.to(self.device)
        self.dim_mapper = self.dim_mapper.to(self.device)

        # Set models to evaluation mode (disables dropout, uses running stats for batchnorm)
        self.efficientnet.eval()
        self.dim_mapper.eval()

    def process_cloth(self, image: np.ndarray) -> Dict:
        """Process cloth image and extract material properties

        This method analyzes a cloth image to determine its properties using
        both deep learning models and traditional computer vision techniques.
        
        Args:
            image: Input cloth image as numpy array (BGR format from OpenCV)

        Returns:
            Dictionary containing cloth analysis results (features, dimensions, contours, etc.)

        Raises:
            ValueError: If input image is None
        """
        if image is None:
            raise ValueError("Input cloth image cannot be None")

        # Make a copy to avoid modifying the original image
        image_copy = image.copy()

        # Convert grayscale to BGR if needed (ensure 3-channel input)
        if len(image_copy.shape) == 2:  # Grayscale
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)

        try:
            # Deep learning-based processing pipeline
            # Preprocess image for model input
            processed_image = self.preprocess_image(image_copy)

            with torch.no_grad():  # Disable gradient calculation for inference
                # Get features from the EfficientNet backbone
                x = self.efficientnet.features(processed_image)
                
                # Global pooling and flatten
                x = self.efficientnet.avgpool(x)
                features = torch.flatten(x, 1)

                # Predict dimensions
                dimensions = self.dim_mapper(features).cpu().numpy()[0]

                # Generate semantic segmentation mask
                segmented_image = self.semantic_segment(processed_image)

            # Cloth detection using a unified approach
            # Start with color-based segmentation (most reliable)
            hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
            
            # Use saturation and value for cloth detection
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            
            # Create a primary mask using color information
            cloth_mask = cv2.bitwise_and(
                cv2.inRange(sat, 20, 255),  # Saturation mask (exclude very unsaturated)
                cv2.inRange(val, 50, 255)   # Value mask (exclude very dark areas)
            )
            
            # Fallback to adaptive thresholding if color-based approach didn't find much
            if np.sum(cloth_mask) < 0.05 * cloth_mask.size:
                gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                cloth_mask = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            
            # Clean up the mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find edges
            edges = cv2.Canny(cloth_mask, 50, 150)

            # Find contours and select the main cloth area
            contours, _ = cv2.findContours(
                cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter small contours
            min_contour_area = 500
            filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
            
            # Process contours and extract information
            if filtered_contours:
                # Use the largest contour as the main cloth
                main_contour = max(filtered_contours, key=cv2.contourArea)
                filtered_contours = [main_contour]
                
                # Create a mask with just the main cloth area
                cloth_area_mask = np.zeros_like(cloth_mask)
                cv2.drawContours(cloth_area_mask, filtered_contours, -1, 255, -1)
                
                # Calculate cloth area
                total_area = cv2.contourArea(filtered_contours[0])
                
                # Get dimensions from the contour
                x, y, w, h = cv2.boundingRect(filtered_contours[0])
                contour_dimensions = np.array([w, h], dtype=np.float32)
                
                # Use contour dimensions if they're reasonable, otherwise use model predictions
                if w > 10 and h > 10:
                    dimensions = contour_dimensions
            else:
                # No significant contours found, use the whole image
                h, w = image_copy.shape[:2]
                cloth_area_mask = np.ones_like(cloth_mask)
                filtered_contours = []
                total_area = w * h

            # Return analysis results
            return {
                "features": features.cpu().numpy(),
                "contours": filtered_contours,
                "dimensions": dimensions,
                "edges": edges,
                "segmented_image": segmented_image.cpu().numpy(),
                "area": total_area,
                "cloth_mask": cloth_area_mask,
            }

        except Exception as e:
            # Fallback to basic image processing if deep learning fails
            print(f"Error during cloth processing: {e}")
            
            # Basic contour detection
            gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Process contours
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                dimensions = np.array([w, h], dtype=np.float32)
                
                cloth_mask = np.zeros_like(gray)
                cv2.drawContours(cloth_mask, [largest_contour], -1, 255, -1)
                
                total_area = cv2.contourArea(largest_contour)
            else:
                h, w = image_copy.shape[:2]
                dimensions = np.array([w, h], dtype=np.float32)
                cloth_mask = np.ones_like(gray) * 255
                total_area = w * h

            # Return simplified results
            return {
                "features": None,
                "contours": contours,
                "dimensions": dimensions,
                "edges": edges,
                "segmented_image": None,
                "area": total_area,
                "cloth_mask": cloth_mask,
                "error": str(e),
            }

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess cloth image for model input"""
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = transform(image).float()

        return image.unsqueeze(0).to(self.device)

    def semantic_segment(self, image: torch.Tensor) -> torch.Tensor:
        """Perform semantic segmentation of the cloth image"""
        with torch.no_grad():
            image = image.to(self.device)
            mask = torch.argmax(self.semantic_segmenter(image), dim=1)
        return mask
