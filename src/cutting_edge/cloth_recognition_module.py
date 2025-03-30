from typing import Dict

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import logging
# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1) # Use updated weights API

        # Modify the classifier head to output the correct number of cloth types
        # The original outputs 1000 classes (ImageNet), we change it to num_cloth_types
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_cloth_types)

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
        efficientnet_features = num_ftrs # Use the actual feature size

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
        self.semantic_segmenter.eval() # Ensure segmenter is also in eval mode
        self.dim_mapper.eval()

    def process_cloth(self, image: np.ndarray) -> Dict:
        """Process cloth image and extract material properties

        This method analyzes a cloth image to determine its properties using
        both deep learning models and traditional computer vision techniques.
        
        Args:
            image: Input cloth image as numpy array (BGR format from OpenCV)

        Returns:
            Dictionary containing cloth analysis results. Returns simplified
            results with an 'error' key if processing fails.

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

        h_orig, w_orig = image_copy.shape[:2]
        
        # --- Initialize default return values ---
        default_dimensions = np.array([w_orig, h_orig], dtype=np.float32)
        results = {
            "features": None,
            "contours": [],
            "dimensions": default_dimensions,
            "edges": np.zeros((h_orig, w_orig), dtype=np.uint8),
            "segmented_image": None,
            "area": 0.0,
            "cloth_mask": np.zeros((h_orig, w_orig), dtype=np.uint8),
            "error": None,
        }

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
                predicted_dimensions = self.dim_mapper(features).cpu().numpy()[0]

                results["features"] = features.cpu().numpy()
                # Store predicted dims, might be overwritten by contour dims later
                results["dimensions"] = predicted_dimensions 

                # Semantic Segmentation
                segmented_output = self.semantic_segment(processed_image)
                # Resize segmentation map back to original size
                # Note: Ensure segmentation output is appropriate (e.g., class indices)
                # This assumes segmented_output is (1, H_proc, W_proc)
                if segmented_output is not None:
                     # Assuming output is class indices, needs conversion for visualization if needed
                     # Resize back to original image dimensions for consistency
                     segmented_map_resized = cv2.resize(
                         segmented_output.cpu().numpy().squeeze(),
                         (w_orig, h_orig),
                         interpolation=cv2.INTER_NEAREST # Use nearest neighbor for class maps
                     )
                     results["segmented_image"] = segmented_map_resized
                else:
                     results["segmented_image"] = None

            # --- Improved Cloth Masking and Contour Detection ---
            gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Slight blur helps Otsu

            # Use Otsu's thresholding + Inverse Binary Threshold
            # This works well for dark objects on light backgrounds
            thresh_val, initial_mask = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            logger.info(f"Otsu threshold value: {thresh_val}")

            # Clean up the mask:
            # MORPH_OPEN removes small noise objects (dots in background)
            # MORPH_CLOSE fills small holes within the main object
            kernel_open = np.ones((5, 5), np.uint8)
            kernel_close = np.ones((7, 7), np.uint8) # Larger kernel for closing
            cleaned_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2) # More closing iterations if needed

            # Find contours on the *cleaned* mask
            # RETR_EXTERNAL finds only the outer contours
            contours, _ = cv2.findContours(
                cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Find the largest contour (should be the cloth)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                min_contour_area = 100 # Adjust if necessary, but Otsu should give a clean main object
                
                if cv2.contourArea(main_contour) > min_contour_area:
                    results["contours"] = [main_contour] # Store only the main contour
                    
                    # Create the final cloth mask by drawing the largest contour
                    final_cloth_mask = np.zeros_like(gray)
                    cv2.drawContours(final_cloth_mask, [main_contour], -1, 255, -1) # Fill the contour
                    results["cloth_mask"] = final_cloth_mask

                    # Calculate area
                    results["area"] = cv2.contourArea(main_contour)

                    # Get dimensions from the contour's bounding box
                    x, y, w, h = cv2.boundingRect(main_contour)
                    contour_dimensions = np.array([w, h], dtype=np.float32)
                    
                    # Use contour dimensions if they seem valid, otherwise keep DL prediction
                    if w > 10 and h > 10: # Basic sanity check
                       logger.info("Using contour dimensions.")
                       results["dimensions"] = contour_dimensions
                    else:
                       logger.warning("Contour dimensions seem small, using predicted dimensions.")
                       # Keep results["dimensions"] as the predicted_dimensions
                
                else:
                    logger.warning("Largest contour area is below threshold. No main cloth detected.")
                    # Keep default empty mask, area=0, etc.
                    # Optionally, could return the 'cleaned_mask' here if partial detection is useful
                    # results["cloth_mask"] = cleaned_mask 
            else:
                logger.warning("No contours found after thresholding and cleaning.")
                # Keep default empty mask, area=0, etc.


            # --- Edge Detection ---
            # Apply Canny to the *final* cloth mask for clearer edges of the detected object
            if np.any(results["cloth_mask"]): # Only run Canny if mask is not empty
                 results["edges"] = cv2.Canny(results["cloth_mask"], 50, 150)
            else:
                 results["edges"] = np.zeros_like(gray) # Ensure edges is correct shape if mask is empty


        except Exception as e:
            logger.error(f"Error during cloth processing: {e}", exc_info=True) # Log traceback
            results["error"] = str(e)
            # Attempt basic fallback using the improved masking?
            try:
                gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh_val, initial_mask = cv2.threshold(
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
                    fallback_mask = np.zeros_like(gray)
                    cv2.drawContours(fallback_mask, [largest_contour], -1, 255, -1)
                    results["cloth_mask"] = fallback_mask
                    results["edges"] = cv2.Canny(fallback_mask, 100, 200)
                else:
                     # If fallback also fails severely, keep defaults but add error
                     results["dimensions"] = default_dimensions # Reset to default dims
            except Exception as fallback_e:
                 logger.error(f"Error during fallback processing: {fallback_e}")
                 results["error"] = f"Main error: {e}; Fallback error: {fallback_e}"
                 # Ensure defaults are set if fallback fails
                 results["dimensions"] = default_dimensions
                 results["contours"] = []
                 results["area"] = 0.0
                 results["cloth_mask"] = np.zeros_like(gray)
                 results["edges"] = np.zeros_like(gray)


        return results

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess cloth image for model input"""
        # Ensure preprocessing matches model training (especially normalization)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((512, 512)), # Resize needed for fixed input models
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] # Standard ImageNet stats
                ),
            ]
        )

        # Convert BGR (OpenCV default) to RGB (PIL/PyTorch default) before ToPILImage
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_tensor = transform(image_rgb).float()

        # Add batch dimension and send to device
        return processed_tensor.unsqueeze(0).to(self.device)

    def semantic_segment(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Perform semantic segmentation of the cloth image"""
        with torch.no_grad():
            # Input tensor should already be preprocessed and on the correct device
            logits = self.semantic_segmenter(image_tensor)
            # Get the class index with the highest probability for each pixel
            mask = torch.argmax(logits, dim=1, keepdim=True) # Keep channel dim for consistency
        # Mask is expected to be on the same device as input (self.device)
        # Return it as is (on GPU/CPU), downstream might move to CPU if needed
        return mask.squeeze(0) # Remove batch dim, keep C=1 H W
