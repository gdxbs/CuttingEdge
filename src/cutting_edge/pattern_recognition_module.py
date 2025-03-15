import os
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from cutting_edge.dataset import DatasetLoader, PatternDataset


class PatternRecognitionModule:
    """Module for pattern recognition and dimension extraction"""

    def __init__(self, dataset_path: str = None, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN for pattern recognition (ResNet50 pretrained)
        #  REF: https://www.youtube.com/watch?v=o_3mboe1jYI&t=3s&pp=ygUIUmVzTmV0NTA%3D
        self.cnn = models.resnet50(pretrained=True)

        # LSTM for corner detection
        # REF: https://www.ej-ai.ejece.org/index.php/ejai/article/view/34
        self.corner_lstm = nn.LSTM(
            input_size=2048, hidden_size=256, num_layers=2, bidirectional=True
        )
        self.dimension_predictor = None

        # Initialize dataset
        self.dataset_loader = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        if dataset_path:
            self._initialize_dataset(dataset_path)

        # Load pretrained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded pretrained model from {model_path}")
        else:
            print(
                "No pretrained model found. Will need training if dataset is provided."
            )

    def _initialize_dataset(self, dataset_path: str):
        """Initialize dataset and modify model architecture accordingly"""

        self.dataset_loader = DatasetLoader(dataset_path)
        self.train_dataset = PatternDataset(self.dataset_loader, "train")
        self.valid_dataset = PatternDataset(self.dataset_loader, "valid")
        self.test_dataset = PatternDataset(self.dataset_loader, "test")

        # Modify model architecture based on dataset
        num_pattern_types = len(self.train_dataset.pattern_types)
        self.cnn.fc = nn.Linear(2048, num_pattern_types)

        # Initialize dimension predictor
        self.dimension_predictor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=32, shuffle=True, num_workers=4
        )
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=32, num_workers=4)

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
                features = self.cnn(images)
                class_output = self.cnn.fc(features)

                # Corner detection
                # feature_seq = features.view(features.size(0), 1, -1)
                # corner_output, _ = self.corner_lstm(feature_seq)

                # Dimension prediction
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

                outputs = self.cnn(images)
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
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

        return transform(image)

    def _extract_contours(self, image: np.ndarray, corners: np.ndarray) -> List:
        """Extract contours using corner information"""

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Refine contours using corner information
        refined_contours = []
        for contour in contours:
            # Check if contour points align with detected corners
            if self._validate_contour_with_corners(contour, corners):
                refined_contours.append(contour)

        return refined_contours

    def extract_dimensions(self, image: np.ndarray) -> Dict[str, float]:
        """Extract dimensions using corner locations"""
        # dimensions = {}
        # # Find the extreme points
        # top = tuple(corners[corners[:, :, 1].argmin()][0])
        # bottom = tuple(corners[corners[:, :, 1].argmax()][0])
        # left = tuple(corners[corners[:, :, 0].argmin()][0])
        # right = tuple(corners[corners[:, :, 0].argmax()][0])

        # # Calculate distances
        # height = np.sqrt((top[0]-bottom[0])**2 + (top[1]-bottom[1])**2)
        # width = np.sqrt((left[0]-right[0])**2 + (left[1]-right[1])**2)

        # dimensions['height'] = height
        # dimensions['width'] = width
        return torch.tensor([256, 256])

    def _validate_contour_with_corners(self, contour: np.ndarray, corners: np.ndarray, threshold=10) -> bool:
        """Validate contour points align with detected corners"""
        if corners is None or len(corners) == 0:
            return True  # No corners to validate against

        # Iterate through each contour point and validate
        for point in contour.squeeze():  # Remove unnecessary dimension from contour
            min_dist = float('inf')

            # Iterate through each corner and see which is closest
            for corner in corners:
                dist = np.sqrt(np.sum((point - corner) ** 2))
                min_dist = min(min_dist, dist)

            if min_dist > threshold:
                return False  # Contour is too far from any corner

        return True  # Contour is valid based on corner proximity

    def process_pattern(self, pattern_image: np.ndarray) -> Dict:
        """Complete pattern processing pipeline"""

        # Preprocess image
        processed_image = self._preprocess_image(pattern_image)
        processed_image = processed_image.to(self.device)

        self.cnn.eval()
        self.corner_lstm.eval()
        if self.dimension_predictor:
            self.dimension_predictor.eval()

        # Extract features using CNN
        with torch.no_grad():
            # Get features
            features = self.cnn(processed_image)

            # Get pattern type if trained with dataset
            if hasattr(self, "train_dataset"):
                class_output = self.cnn.fc(features)
                pattern_type = torch.argmax(class_output, dim=1).item()
                pattern_type_name = self.train_dataset.pattern_types[pattern_type]
            else:
                pattern_type_name = None

            # Get corners
            feature_seq = features.view(features.size(0), 1, -1)
            corner_output, _ = self.corner_lstm(feature_seq)
            corners = torch.sigmoid(corner_output).cpu().numpy()

            # Get dimensions
            if self.dimension_predictor:
                dimensions = self.dimension_predictor(features).cpu().numpy()
            else:
                dimensions = self.extract_dimensions(pattern_image)

            # Extract contours
            contours = self._extract_contours(pattern_image, corners)

        return {
            "pattern_type": pattern_type_name,
            "corners": corners,
            "dimensions": dimensions,
            "contours": contours,
            "features": features.cpu().numpy(),
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