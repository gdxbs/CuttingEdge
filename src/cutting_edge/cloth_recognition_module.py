from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp
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

    def __init__(self, num_cloth_types=10, encoder_name="resnet34", encoder_weights="imagenet"):
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
        self.efficientnet = models.efficientnet_b0(pretrained=True)  # Pretrained on ImageNet-1k
        
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
            encoder_name=encoder_name,       # ResNet34 provides good accuracy/speed balance
            encoder_weights=encoder_weights, # Using ImageNet pretrained weights
            in_channels=3,                   # RGB input images
            classes=num_cloth_types,         # Output channels match cloth types
        ).to(self.device)

        # Dimension mapping network (MLP) for estimating cloth dimensions
        # Takes EfficientNet features (1000-dim) and predicts width/height (2-dim)
        self.dim_mapper = nn.Sequential(
            nn.Linear(1000, 512),  # First layer reduces features to 512 dimensions
            nn.ReLU(),             # ReLU activation for non-linearity
            nn.Linear(512, 2),     # Final layer outputs width and height
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
                # Extract features using EfficientNet
                features = self.efficientnet(processed_image)

                # Predict cloth dimensions using MLP
                dimensions = self.dim_mapper(features).cpu().numpy()[0]

                # Generate semantic segmentation mask
                segmented_image = self.semantic_segment(processed_image)

            # Traditional computer vision processing for contour detection
            # This approach is more robust for detecting cloth boundaries
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise (5x5 kernel)
            # This helps improve edge detection by reducing image noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Use adaptive thresholding for better edge detection on various fabric textures
            # Adaptive threshold adjusts according to local image regions, which works better
            # for cloth materials with varying textures and lighting conditions
            # Parameters: (src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
            # REF: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Edge detection using Canny algorithm
            # Threshold values 100, 200 provide a good balance for cloth edges
            # REF: "A Computational Approach to Edge Detection" (Canny, 1986)
            edges = cv2.Canny(thresh, 100, 200)

            # Dilate edges to connect broken contours
            # This helps create continuous boundaries for contour detection
            kernel = np.ones((3, 3), np.uint8)  # 3x3 structuring element
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            # Find contours in the processed edge map
            # RETR_EXTERNAL retrieves only the outermost contours
            # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
            contours, _ = cv2.findContours(
                dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter small contours that likely represent noise
            # Contours smaller than 100 pixels in area are considered noise
            min_contour_area = 100  # Minimum area threshold in square pixels
            filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

            # Calculate total cloth area by summing contour areas
            total_area = 0
            for contour in filtered_contours:
                total_area += cv2.contourArea(contour)

            # Log analysis results for debugging
            print(f"Cloth analysis complete: Found {len(filtered_contours)} significant contours")
            print(f"Cloth dimensions: {dimensions}, Total area: {total_area} square pixels")

            # Return comprehensive analysis results
            return {
                "features": features.cpu().numpy(),         # Neural network features
                "contours": filtered_contours,            # Detected cloth contours
                "dimensions": dimensions,                 # Estimated dimensions (width, height)
                "edges": edges,                          # Edge detection result
                "segmented_image": segmented_image.cpu().numpy(),  # Semantic segmentation
                "area": total_area                       # Total cloth area in square pixels
            }

        except Exception as e:
            # Error handling with fallback to traditional CV methods
            print(f"Error during cloth processing: {e}")

            # Fallback using basic contour detection when deep learning fails
            # This ensures we still provide useful results even if models fail
            gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)  # Standard Canny edge parameters
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Estimate dimensions based on contour bounding box or image size
            if contours:
                # Use the largest contour (by area) to determine cloth dimensions
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)  # Get bounding rectangle
                dimensions = np.array([w, h], dtype=np.float32)  # Width and height
            else:
                # If no contours found, use the entire image dimensions as fallback
                h, w = image_copy.shape[:2]  # Image height and width
                dimensions = np.array([w, h], dtype=np.float32)

            # Return basic analysis with error information
            return {
                "features": None,              # No neural features available
                "contours": contours,        # Basic contours from Canny
                "dimensions": dimensions,    # Estimated dimensions
                "edges": edges,              # Basic edge detection result
                "segmented_image": None,     # No segmentation available
                "error": str(e)              # Error message for debugging
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