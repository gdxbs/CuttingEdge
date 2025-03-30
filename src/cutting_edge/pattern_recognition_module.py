import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from cutting_edge.dataset import DatasetLoader, PatternDataset


class PatternCNN(nn.Module):
    """Pattern CNN using ResNet50 as backbone for feature extraction

    This class implements a CNN using ResNet50 as the backbone for feature extraction.
    It replaces the final fully connected layer with a new one that outputs the number of classes
    specified by the user. The model is designed to be used for pattern recognition tasks.
    """

    def __init__(self, num_classes, pretrained=True):  # Added num_classes parameter
        super().__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        # Use all layers except the last FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add our own FC layer with the specified number of classes
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Get features from ResNet
        features = self.features(x)  # Should be [batch_size, 2048, 1, 1]
        features_flat = torch.flatten(features, 1)  # Flatten to [batch_size, 2048]
        # Pass through FC layer
        x = self.fc(features_flat)
        return x, features_flat


class PatternRecognitionModule:
    """Module for garment pattern recognition and dimension extraction

    This module implements a multi-task deep learning system for analyzing garment patterns.
    It performs three main tasks:
    1. Pattern type classification (e.g., shirt, pants, dress)
    2. Corner point detection for pattern structure analysis
    3. Pattern dimension estimation

    The architecture combines a CNN backbone (ResNet50) for feature extraction with
    task-specific heads for classification, corner detection (LSTM), and dimension prediction.

    References:
    - ResNet architecture: "Deep Residual Learning for Image Recognition" (He et al., 2016)
      https://arxiv.org/abs/1512.03385
    - Corner detection approach adapted from: "Garment Pattern Recognition using Deep Learning"
      (Chen & Wang, 2023)
      https://www.ej-ai.ejece.org/index.php/ejai/article/view/34
    """

    def __init__(
        self, dataset_path: Optional[str] = None, model_path: Optional[str] = None
    ):
        """Initialize the pattern recognition module

        Args:
            dataset_path: Path to the GarmentCodeData dataset (optional)
            model_path: Path to a pretrained model checkpoint (optional)
        """
        # Set the computation device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN feature extractor using ResNet50 architecture pretrained on ImageNet
        # ResNet is effective for pattern recognition due to its ability to learn
        # hierarchical features while addressing the vanishing gradient problem
        # REF: "Deep Residual Learning for Image Recognition" (He et al., 2016)
        # https://arxiv.org/abs/1512.03385
        # Initialize with 1 class as default (will be updated in _initialize_dataset if dataset provided)
        self.cnn = PatternCNN(num_classes=1, pretrained=True)

        # LSTM for corner detection (sequence modeling of pattern structure)
        # Bidirectional LSTM processes feature sequences to detect pattern corners
        # The architecture follows the approach described in:
        # REF: "Garment Pattern Recognition using Deep Learning" (Chen & Wang, 2023)
        # https://www.ej-ai.ejece.org/index.php/ejai/article/view/34
        self.corner_lstm = nn.LSTM(
            input_size=2048,  # Input size matches ResNet50 feature dimension
            hidden_size=256,  # Hidden state dimension
            num_layers=2,  # Number of LSTM layers for better sequence modeling
            bidirectional=True,  # Bidirectional to consider context in both directions
        )
        self.dimension_predictor = None  # Will be initialized based on dataset

        # Initialize dataset components (populated in _initialize_dataset if path provided)
        self.dataset_loader = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        if dataset_path:
            self._initialize_dataset(dataset_path)

        # Load pretrained model weights if provided and exists
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded pretrained model from {model_path}")
        else:
            print(
                "No pretrained model found. Will need training if dataset is provided."
            )

    def _initialize_dataset(self, dataset_path: str):
        """Initialize dataset and modify model architecture accordingly

        This method sets up the dataset and adapts the model architecture based on
        the dataset characteristics (e.g., number of pattern types).

        Args:
            dataset_path: Path to the GarmentCodeData dataset
        """

        # Initialize dataset components
        self.dataset_loader = DatasetLoader(dataset_path)
        self.train_dataset = PatternDataset(self.dataset_loader, "training")
        self.valid_dataset = PatternDataset(self.dataset_loader, "validation")
        self.test_dataset = PatternDataset(self.dataset_loader, "test")

        # Modify classification head based on number of pattern types in dataset
        # Replace the final fully connected layer of ResNet50 with a new one
        # that outputs the correct number of classes for our specific dataset
        num_pattern_types = len(self.train_dataset.pattern_types)
        self.cnn = PatternCNN(num_classes=num_pattern_types)

        # Initialize dimension predictor network
        # This MLP takes ResNet features and predicts pattern dimensions (width, height)
        # Architecture: 3-layer MLP with ReLU activations and decreasing width
        # Following recommendations from:
        # "Delving Deep into Rectifiers" (He et al., 2015)
        # REF: https://arxiv.org/abs/1502.01852
        self.dimension_predictor = nn.Sequential(
            nn.Linear(2048, 512),  # First layer reduces dimension from 2048 to 512
            nn.ReLU(),  # ReLU activation for non-linearity
            nn.Linear(512, 256),  # Second layer further reduces to 256
            nn.ReLU(),  # Another ReLU activation
            nn.Linear(256, 2),  # Output layer for width and height prediction
        )

        # Create PyTorch DataLoader instances for efficient batch processing
        # - batch_size=32: Standard mini-batch size, balances memory usage and parallelism
        # - shuffle=True: Randomizes data order for training to improve generalization
        # - num_workers=4: Uses multiple CPU threads for data loading

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=7,
            pin_memory=True,
            prefetch_factor=2,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=64,
            num_workers=7,
            pin_memory=True,
            prefetch_factor=2,
        )

        # Move models to the selected computation device (GPU/CPU)
        self.cnn = self.cnn.to(self.device)
        self.corner_lstm = self.corner_lstm.to(self.device)
        self.dimension_predictor = self.dimension_predictor.to(self.device)

        # Initialize optimizers for each network component
        # Adam optimizer with default parameters (lr=1e-3)
        # REF: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2015)
        # https://arxiv.org/abs/1412.6980
        self.cnn_optimizer = optim.Adam(self.cnn.parameters())
        self.lstm_optimizer = optim.Adam(self.corner_lstm.parameters())
        self.dim_optimizer = optim.Adam(self.dimension_predictor.parameters())

        # Loss functions for each task
        self.classification_loss = nn.CrossEntropyLoss()  # Standard for classification
        self.corner_loss = nn.BCEWithLogitsLoss()  # For binary corner detection
        self.dimension_loss = nn.MSELoss()  # For regression of dimensions

        # Training tracking variables
        self.best_accuracy = 0  # Tracks best validation accuracy
        self.best_model_state = None  # Stores best model checkpoint

    def check_and_train(self, model_path: str, num_epochs: int = 50):
        """Check if model exists, train if it doesn't"""

        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            self.load_model(model_path)
        else:
            if self.dataset_loader is None:
                raise ValueError(
                    "Cannot train without dataset. Please initialize with dataset_path."
                )

            print(f"No existing model found at {model_path}. Starting training...")
            self.train(num_epochs)
            self.save_model(model_path)
            print(f"Model trained and saved to {model_path}")

    def train(self, num_epochs: int):
        """Train all components of the pattern recognition module"""

        if self.dataset_loader is None:
            raise ValueError(
                "Cannot train without dataset. Please initialize with dataset_path."
            )

        for epoch in range(num_epochs):
            self.cnn.train()
            self.corner_lstm.train()
            self.dimension_predictor.train()

            total_loss = 0
            correct_predictions = 0
            total_samples = 0
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                dimensions = batch["dimensions"].to(self.device)
                # corner_maps = batch["corner_maps"].to(self.device)

                # Forward pass through CNN
                ## features = self.cnn(images)
                ## print(features.shape)
                ## features = features.view(features.size(0), -1)
                ## print(features.shape)
                ## print(images, labels, dimensions, features)
                ## class_output = self.cnn.fc(features)
                ## print(class_output)
                class_output, features = self.cnn(images)
                # Corner detection
                # feature_seq = features.view(features.size(0), 1, -1)
                # corner_output, _ = self.corner_lstm(feature_seq)

                # Dimension prediction using features
                dim_output = self.dimension_predictor(features)

                # Calculate losses
                class_loss = self.classification_loss(class_output, labels)
                # corner_loss = self.corner_loss(corner_output, corner_maps) # you dont need the corner maps when training
                dim_loss = self.dimension_loss(dim_output, dimensions)

                # Combined loss
                # total_batch_loss = class_loss + corner_loss + dim_loss # you dont need the corner maps when training
                total_batch_loss = class_loss + dim_loss

                # Backward pass
                self.cnn_optimizer.zero_grad()
                self.lstm_optimizer.zero_grad()
                self.dim_optimizer.zero_grad()

                total_batch_loss.backward()

                self.cnn_optimizer.step()
                self.lstm_optimizer.step()
                self.dim_optimizer.step()

                # Update metrics
                total_loss += total_batch_loss.item()
                _, predicted = class_output.max(1)
                correct_predictions += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch: {epoch}, Batch: {batch_idx}, "
                        f"Loss: {total_batch_loss.item():.4f}"
                    )

            # Validation phase
            val_accuracy = self.validate()
            print(f"Validation Accuracy: {100.*val_accuracy:.2f}%")

            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_model_state = {
                    "cnn": self.cnn.state_dict(),
                    "lstm": self.corner_lstm.state_dict(),
                    "dim_predictor": (
                        self.dimension_predictor.state_dict()
                        if self.dimension_predictor
                        else None
                    ),
                    "accuracy": val_accuracy,
                    "epoch": epoch,
                }

            print(
                f"Epoch {epoch} completed. "
                f"Training Accuracy: {100.*correct_predictions/total_samples:.2f}%, "
                f"Validation Accuracy: {100.*val_accuracy:.2f}%"
            )

    def validate(self) -> float:
        """Add validation method"""

        self.cnn.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.valid_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs, _ = self.cnn(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return correct / total

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.Grayscale(
                    num_output_channels=3
                ),  # Convert to 3 channels to match ResNet input
                transforms.ToTensor(),
            ]
        )

        return transform(image)

    def _extract_contours(self, image: np.ndarray, corners: np.ndarray) -> List:
        """Extract contours focusing on the actual pattern area
        
        Enhanced implementation that uses multiple detection methods in priority order:
        1. Color-based segmentation for better pattern isolation
        2. Multiple thresholding techniques to isolate patterns
        3. Corner-based contour creation (if corners available)
        4. Partial rectangle fallback (not full image) if no other methods succeed
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # First approach: Multiple color thresholding to better detect separate pattern pieces
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract saturation channel which often separates pattern from background
        sat = hsv[:, :, 1]
        
        # Try with RETR_EXTERNAL first (for outer contours of pattern pieces)
        _, binary = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # If not enough pixels detected (under 5%), try on grayscale with inverse
        if np.sum(binary) / binary.size < 0.05:
            # Try inverse binary threshold on grayscale with Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up binary image with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill holes
        
        # Try to find contours with tree hierarchy to detect nested pattern pieces
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Try RETR_EXTERNAL approach only if we find too many or no contours
        if len(contours) > 10 or len(contours) == 0:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out small contours
        min_area_ratio = 0.01  # Minimum area as percentage of image
        min_area = min_area_ratio * gray.shape[0] * gray.shape[1]
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Try a different approach if we still don't have enough contours
        if len(filtered_contours) < 2:
            # Try a secondary approach with Canny edge detection
            edges = cv2.Canny(gray, 30, 150) 
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            additional_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            # Add these to our filtered contours
            filtered_contours.extend(additional_contours)
            
            # Remove duplicates by comparing centroids
            unique_contours = []
            centroids = []
            
            for contour in filtered_contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                # Check if this centroid is already in our list
                is_duplicate = False
                for existing_cx, existing_cy in centroids:
                    if abs(cx - existing_cx) < 20 and abs(cy - existing_cy) < 20:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    centroids.append((cx, cy))
                    unique_contours.append(contour)
                    
            filtered_contours = unique_contours
        
        # If we found valid contours, return them (keeping up to 3 largest contours)
        if filtered_contours:
            # Sort by size (largest first) and keep up to 3 largest contours
            sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
            return sorted_contours[:3]
        
        # If nothing worked, use adaptive thresholding as a fallback
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)
        binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if filtered_contours:
            # Sort by size and keep up to 3 largest contours
            sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
            return sorted_contours[:3]
        
        # Last resort: If corners are available, create a convex hull
        if corners is not None and len(corners) > 0:
            try:
                corners_array = np.array(corners).reshape(-1, 2)
                if len(corners_array) >= 4:
                    hull = cv2.convexHull(corners_array.astype(np.float32))
                    return [hull]
            except Exception as e:
                print(f"Error creating contour from corners: {e}")
        
        # If all else fails, return a smaller central rectangle (80% of image size)
        # rather than the full image rectangle
        h, w = gray.shape[:2]
        margin_x, margin_y = int(w * 0.1), int(h * 0.1)
        simple_contour = np.array([
            [[margin_x, margin_y]], 
            [[w - margin_x, margin_y]], 
            [[w - margin_x, h - margin_y]], 
            [[margin_x, h - margin_y]]
        ], dtype=np.int32)
        
        return [simple_contour]

    def extract_dimensions(self, image: np.ndarray, corners=None) -> torch.Tensor:
        """Extract dimensions using corner locations or image properties"""
        if corners is not None and len(corners) > 0:
            # Find the extreme points if corners are provided
            try:
                # Ensure corners is a proper numpy array with the right shape
                corners_array = np.array(corners).reshape(-1, 2)

                # Find extreme points
                top = corners_array[corners_array[:, 1].argmin()]
                bottom = corners_array[corners_array[:, 1].argmax()]
                left = corners_array[corners_array[:, 0].argmin()]
                right = corners_array[corners_array[:, 0].argmax()]

                # Calculate distances in pixels
                height = np.sqrt((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2)
                width = np.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)

                return torch.tensor([width, height], dtype=torch.float32)
            except (IndexError, ValueError) as e:
                # Log the error but continue with fallback method
                print(f"Error extracting dimensions from corners: {e}")

        # Fallback: Use image dimensions with aspect ratio preserved
        h, w = image.shape[:2]
        # Scale to reasonable pattern size while maintaining aspect ratio
        scale = min(256 / w, 256 / h)
        width = w * scale
        height = h * scale

        return torch.tensor([width, height], dtype=torch.float32)

    def _validate_contour_with_corners(
        self, contour: np.ndarray, corners: np.ndarray, threshold=10
    ) -> bool:
        """Validate contour points align with detected corners"""
        if corners is None or len(corners) == 0:
            return True  # No corners to validate against

        # Iterate through each contour point and validate
        for point in contour.squeeze():  # Remove unnecessary dimension from contour
            min_dist = float("inf")

            # Iterate through each corner and see which is closest
            for corner in corners:
                dist = np.sqrt(np.sum((point - corner) ** 2))
                min_dist = min(min_dist, dist)

            if min_dist > threshold:
                return False  # Contour is too far from any corner

        return True  # Contour is valid based on corner proximity

    def process_pattern(self, pattern_image: np.ndarray) -> Dict:
        """Complete pattern processing pipeline"""
        if pattern_image is None:
            raise ValueError("Input pattern image cannot be None")

        # Make a copy to avoid modifying the original
        pattern_image_copy = pattern_image.copy()

        # Check image format
        if len(pattern_image_copy.shape) == 2:  # Grayscale
            pattern_image_copy = cv2.cvtColor(pattern_image_copy, cv2.COLOR_GRAY2BGR)

        # Preprocess image
        processed_image = self._preprocess_image(pattern_image_copy)
        processed_image = processed_image.unsqueeze(0).to(
            self.device
        )  # Add batch dimension

        # Set models to evaluation mode
        self.cnn.eval()
        self.corner_lstm.eval()
        if self.dimension_predictor:
            self.dimension_predictor.eval()

        # Extract features using CNN
        with torch.no_grad():
            try:
                # Get features
                class_output, features = self.cnn(processed_image)

                # Get pattern type if trained with dataset
                if hasattr(self, "train_dataset") and hasattr(
                    self.train_dataset, "pattern_types"
                ):
                    pattern_type = torch.argmax(class_output, dim=1).item()
                    pattern_type_name = self.train_dataset.pattern_types[pattern_type]
                else:
                    pattern_type_name = (
                        "unknown"  # Default if not trained or no types available
                    )

                # Get corners using flattened features for consistent dimensions
                feature_seq = features.view(features.size(0), 1, -1)
                corner_output, _ = self.corner_lstm(feature_seq)
                corners = torch.sigmoid(corner_output).cpu().numpy().squeeze()

                # Get dimensions
                if self.dimension_predictor:
                    dimensions = (
                        self.dimension_predictor(features).cpu().numpy().squeeze()
                    )
                else:
                    dimensions = self.extract_dimensions(pattern_image_copy, corners)

                # Extract contours
                contours = self._extract_contours(pattern_image_copy, corners)

                # Debug log
                print(
                    f"Pattern analysis complete: Type={pattern_type_name}, Dimensions={dimensions}, Found {len(contours)} pattern pieces"
                )

                return {
                    "pattern_type": pattern_type_name,
                    "corners": corners,
                    "dimensions": dimensions,
                    "contours": contours,
                    "features": features.cpu().numpy().squeeze(),
                    "num_pattern_pieces": len(contours)
                }

            except Exception as e:
                print(f"Error during pattern processing: {e}")

                # Fallback: Return basic information using traditional CV methods
                # Extract edges and contours using basic CV
                gray = cv2.cvtColor(pattern_image_copy, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Check if contours are empty and create a minimal contour if needed
                if len(contours) == 0:
                    print(
                        "No contours found in pattern, creating a simple rectangular contour"
                    )
                    h, w = pattern_image_copy.shape[:2]
                    simple_contour = np.array(
                        [[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32
                    )
                    contours = [simple_contour]

                # Get basic dimensions
                dimensions = self.extract_dimensions(pattern_image_copy)

                return {
                    "pattern_type": "unknown",
                    "corners": None,
                    "dimensions": dimensions,
                    "contours": contours,
                    "features": None,
                    "error": str(e),
                }

    def save_model(self, model_path: str):
        """Save the model weights to the given path"""
        torch.save(self.best_model_state, model_path)

    def load_model(self, model_path: str):
        """Load model weights from the given path"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint["cnn"])
        self.corner_lstm.load_state_dict(checkpoint["lstm"])
        if self.dimension_predictor:
            self.dimension_predictor.load_state_dict(checkpoint["dim_predictor"])
        self.best_accuracy = checkpoint["accuracy"]
