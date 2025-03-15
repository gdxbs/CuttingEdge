from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms


class ClothRecognitionModule:
    """Module for cloth recognition and dimension mapping"""

    def __init__(self, num_cloth_types=10, encoder_name="resnet34", encoder_weights="imagenet"):  # Set Default Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cloth Recognition CNN
        # Trained on ImageNet-1k
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        # Modify the classifier to match the number of cloth types
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, num_cloth_types
        )
        # Semantic Segmentation Model
        self.semantic_segmenter = smp.Unet(  # Or other model like DeepLabV3
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_cloth_types,  # Ensure this aligns with num_cloth_types
        ).to(self.device)

        # Additional layers for dimension mapping
        self.dim_mapper = nn.Sequential(
            nn.Linear(1000, 512),  # Adjusted input dimension to 1000
            nn.ReLU(),
            nn.Linear(512, 2),  # width, height
        )

        # Move models to device
        self.efficientnet = self.efficientnet.to(self.device)
        self.dim_mapper = self.dim_mapper.to(self.device)

        # Set models to evaluation mode
        self.efficientnet.eval()
        self.dim_mapper.eval()

    def process_cloth(self, image: np.ndarray) -> Dict:
        """Process cloth image and extract properties"""
        # Preprocess
        processed_image = self.preprocess_image(image)

        with torch.no_grad():
            # Extract features
            features = self.efficientnet(processed_image)

            # Predict dimensions
            dimensions = self.dim_mapper(features).cpu().numpy()[0]

            # Semantic Segmentation
            segmented_image = self.semantic_segment(processed_image)

        # Detect contours on original image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)  # Consider adaptive thresholding
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return {
            "features": features.cpu().numpy(),
            "contours": contours,
            "dimensions": dimensions,
            "edges": edges,
            "segmented_image": segmented_image.cpu().numpy(),
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