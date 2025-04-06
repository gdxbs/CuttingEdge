import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from cutting_edge.dataset import DatasetLoader, PatternDataset
from cutting_edge.utils import preprocess_image_for_model, extract_contours


class PatternCNN(nn.Module):
    """Pattern CNN for feature extraction and classification
    
    Uses ResNet50 as backbone for feature extraction and adds
    a classification layer for pattern types.
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        # Use all layers except the last FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add our own FC layer
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Extract features
        features = self.features(x)
        features_flat = torch.flatten(features, 1)
        # Pass through FC layer
        x = self.fc(features_flat)
        return x, features_flat


class PatternRecognitionModule:
    """Module for garment pattern recognition and dimension extraction
    
    This module analyzes garment patterns in images to:
    1. Classify pattern type (e.g., shirt, pants, dress)
    2. Detect corner points for structure analysis
    3. Estimate pattern dimensions
    """

    def __init__(self, dataset_path: Optional[str] = None, model_path: Optional[str] = None):
        """Initialize the pattern recognition module"""
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize CNN with default single class
        self.cnn = PatternCNN(num_classes=1, pretrained=True)

        # LSTM for corner detection
        self.corner_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
        )
        
        # The dimension predictor will be initialized based on the dataset
        self.dimension_predictor = None

        # Initialize dataset components if path provided
        self.dataset_loader = None
        self.train_loader = None
        self.valid_loader = None
        
        if dataset_path:
            self._initialize_dataset(dataset_path)

        # Load pretrained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded pretrained model from {model_path}")
        else:
            print("No pretrained model found. Will need training if dataset is provided.")

    def _initialize_dataset(self, dataset_path: str):
        """Initialize dataset and modify model architecture"""
        # Load dataset
        self.dataset_loader = DatasetLoader(dataset_path)
        self.train_dataset = PatternDataset(self.dataset_loader, "training")
        self.valid_dataset = PatternDataset(self.dataset_loader, "validation")

        # Update CNN to have the correct number of classes
        num_pattern_types = len(self.train_dataset.pattern_types)
        self.cnn = PatternCNN(num_classes=num_pattern_types)

        # Initialize dimension predictor
        self.dimension_predictor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # Width and height
        )

        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=64,
            num_workers=4,
            pin_memory=True,
        )

        # Move models to device
        self.cnn = self.cnn.to(self.device)
        self.corner_lstm = self.corner_lstm.to(self.device)
        self.dimension_predictor = self.dimension_predictor.to(self.device)

        # Initialize optimizers
        self.cnn_optimizer = optim.Adam(self.cnn.parameters())
        self.lstm_optimizer = optim.Adam(self.corner_lstm.parameters())
        self.dim_optimizer = optim.Adam(self.dimension_predictor.parameters())

        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.corner_loss = nn.BCEWithLogitsLoss()
        self.dimension_loss = nn.MSELoss()

        # Training tracking
        self.best_accuracy = 0
        self.best_model_state = None

    def check_and_train(self, model_path: str, num_epochs: int = 50):
        """Check if model exists, train if it doesn't"""
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            self.load_model(model_path)
        else:
            if self.dataset_loader is None:
                raise ValueError("Cannot train without dataset. Please initialize with dataset_path.")

            print(f"No existing model found at {model_path}. Starting training...")
            self.train(num_epochs)
            self.save_model(model_path)
            print(f"Model trained and saved to {model_path}")

    def train(self, num_epochs: int):
        """Train all components of the module"""
        if self.dataset_loader is None:
            raise ValueError("Cannot train without dataset. Please initialize with dataset_path.")

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

                # Forward pass through CNN
                class_output, features = self.cnn(images)
                
                # Dimension prediction
                dim_output = self.dimension_predictor(features)

                # Calculate losses
                class_loss = self.classification_loss(class_output, labels)
                dim_loss = self.dimension_loss(dim_output, dimensions)
                total_batch_loss = class_loss + dim_loss

                # Backward pass
                self.cnn_optimizer.zero_grad()
                self.lstm_optimizer.zero_grad()
                self.dim_optimizer.zero_grad()
                total_batch_loss.backward()

                # Update weights
                self.cnn_optimizer.step()
                self.lstm_optimizer.step()
                self.dim_optimizer.step()

                # Update metrics
                total_loss += total_batch_loss.item()
                _, predicted = class_output.max(1)
                correct_predictions += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)

                if batch_idx % 100 == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {total_batch_loss.item():.4f}")

            # Validation phase
            val_accuracy = self.validate()
            print(f"Validation Accuracy: {100.*val_accuracy:.2f}%")

            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_model_state = {
                    "cnn": self.cnn.state_dict(),
                    "lstm": self.corner_lstm.state_dict(),
                    "dim_predictor": self.dimension_predictor.state_dict() if self.dimension_predictor else None,
                    "accuracy": val_accuracy,
                    "epoch": epoch,
                }

            print(
                f"Epoch {epoch} completed. "
                f"Training Accuracy: {100.*correct_predictions/total_samples:.2f}%, "
                f"Validation Accuracy: {100.*val_accuracy:.2f}%"
            )

    def validate(self) -> float:
        """Validate the model"""
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

    def extract_dimensions(self, image: np.ndarray, corners=None) -> torch.Tensor:
        """Extract dimensions from pattern image"""
        if corners is not None and len(corners) > 0:
            try:
                # Ensure corners is a proper numpy array
                corners_array = np.array(corners).reshape(-1, 2)
                
                # Find extreme points
                top = corners_array[corners_array[:, 1].argmin()]
                bottom = corners_array[corners_array[:, 1].argmax()]
                left = corners_array[corners_array[:, 0].argmin()]
                right = corners_array[corners_array[:, 0].argmax()]
                
                # Calculate distances
                height = np.sqrt((top[0] - bottom[0])**2 + (top[1] - bottom[1])**2)
                width = np.sqrt((left[0] - right[0])**2 + (left[1] - right[1])**2)
                
                return torch.tensor([width, height], dtype=torch.float32)
            except Exception as e:
                print(f"Error extracting dimensions from corners: {e}")
        
        # Fallback: Use image dimensions with aspect ratio preserved
        h, w = image.shape[:2]
        scale = min(256 / w, 256 / h)
        width = w * scale
        height = h * scale
        
        return torch.tensor([width, height], dtype=torch.float32)

    def process_pattern(self, pattern_image: np.ndarray) -> Dict:
        """Process a pattern image and extract features"""
        if pattern_image is None:
            raise ValueError("Input pattern image cannot be None")
        
        # Make a copy to avoid modifying the original
        pattern_image_copy = pattern_image.copy()
        
        # Ensure we have a 3-channel image
        if len(pattern_image_copy.shape) == 2:
            pattern_image_copy = cv2.cvtColor(pattern_image_copy, cv2.COLOR_GRAY2BGR)
        
        # Set models to evaluation mode
        self.cnn.eval()
        self.corner_lstm.eval()
        if self.dimension_predictor:
            self.dimension_predictor.eval()
        
        # Extract features
        try:
            # Preprocess image
            processed_image = preprocess_image_for_model(pattern_image_copy, self.device)
            
            with torch.no_grad():
                # Get features
                class_output, features = self.cnn(processed_image)
                
                # Get pattern type if trained with dataset
                if hasattr(self, "train_dataset") and hasattr(self.train_dataset, "pattern_types"):
                    pattern_type = torch.argmax(class_output, dim=1).item()
                    pattern_type_name = self.train_dataset.pattern_types[pattern_type]
                else:
                    pattern_type_name = "unknown"
                
                # Get corners
                feature_seq = features.view(features.size(0), 1, -1)
                corner_output, _ = self.corner_lstm(feature_seq)
                corners = torch.sigmoid(corner_output).cpu().numpy().squeeze()
                
                # Get dimensions
                if self.dimension_predictor:
                    dimensions = self.dimension_predictor(features).cpu().numpy().squeeze()
                else:
                    dimensions = self.extract_dimensions(pattern_image_copy, corners)
                
                # Extract contours using shared utility
                contours = extract_contours(pattern_image_copy)
                
                return {
                    "pattern_type": pattern_type_name,
                    "corners": corners,
                    "dimensions": dimensions,
                    "contours": contours,
                    "features": features.cpu().numpy().squeeze(),
                    "num_pattern_pieces": len(contours),
                }
            
        except Exception as e:
            print(f"Error during pattern processing: {e}")
            
            # Fallback using traditional CV methods
            gray = cv2.cvtColor(pattern_image_copy, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a minimal contour if none found
            if len(contours) == 0:
                h, w = pattern_image_copy.shape[:2]
                simple_contour = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)
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
        """Save the model weights"""
        torch.save(self.best_model_state, model_path)

    def load_model(self, model_path: str):
        """Load model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint["cnn"])
        self.corner_lstm.load_state_dict(checkpoint["lstm"])
        if self.dimension_predictor and checkpoint["dim_predictor"]:
            self.dimension_predictor.load_state_dict(checkpoint["dim_predictor"])
        self.best_accuracy = checkpoint["accuracy"]