import os
from typing import Dict, Optional, List, Tuple
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cutting_edge.config import (
    DATASET,
    IMAGE_PROCESSING,
    MODEL,
    TRAINING,
    VISUALIZATION,
    AUGMENTATION,
)
from cutting_edge.dataset import DatasetLoader, PatternDataset
from cutting_edge.utils import extract_contours, preprocess_image_for_model


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
        self.fc = nn.Linear(MODEL["DEFAULT_FLATTENED_SIZE"], num_classes)

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

    def __init__(
        self, dataset_path: Optional[str] = None, model_path: Optional[str] = None
    ):
        """Initialize the pattern recognition module"""
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize CNN with default single class
        self.cnn = PatternCNN(num_classes=1, pretrained=True)

        # LSTM for corner detection
        self.corner_lstm = nn.LSTM(
            input_size=MODEL["DEFAULT_FLATTENED_SIZE"],
            hidden_size=MODEL["LSTM_HIDDEN_SIZE"],
            num_layers=MODEL["LSTM_NUM_LAYERS"],
            bidirectional=MODEL["LSTM_BIDIRECTIONAL"],
        )

        # The dimension predictor will be initialized based on the dataset
        self.dimension_predictor = None

        # Initialize dataset components if path provided
        self.dataset_loader = None
        self.train_loader = None
        self.valid_loader = None

        # Initialize schedulers
        self.cnn_scheduler = None
        self.lstm_scheduler = None
        self.dim_scheduler = None

        # Early stopping parameters
        self.early_stopping_patience = TRAINING["EARLY_STOPPING_PATIENCE"]
        self.early_stopping_counter = 0
        self.early_stopping_min_delta = TRAINING["EARLY_STOPPING_MIN_DELTA"]

        # Checkpoint directory
        self.checkpoint_dir = TRAINING["CHECKPOINT_DIR"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
        """Initialize dataset and modify model architecture"""
        # Load dataset
        self.dataset_loader = DatasetLoader(dataset_path)
        self.train_dataset = PatternDataset(self.dataset_loader, "training")
        self.valid_dataset = PatternDataset(self.dataset_loader, "validation")

        # Update CNN to have the correct number of classes
        num_pattern_types = len(self.train_dataset.pattern_types)
        self.cnn = PatternCNN(num_classes=num_pattern_types)

        # Initialize dimension predictor with batch normalization and dropout
        self.dimension_predictor = nn.Sequential(
            nn.Linear(MODEL["DEFAULT_FLATTENED_SIZE"], MODEL["HIDDEN_DIM"]),
            nn.BatchNorm1d(MODEL["HIDDEN_DIM"]),
            nn.ReLU(),
            nn.Dropout(MODEL["DIMENSION_PREDICTOR_DROPOUT_1"]),
            nn.Linear(MODEL["HIDDEN_DIM"], MODEL["HIDDEN_DIM"] // 2),
            nn.BatchNorm1d(MODEL["HIDDEN_DIM"] // 2),
            nn.ReLU(),
            nn.Dropout(MODEL["DIMENSION_PREDICTOR_DROPOUT_2"]),
            nn.Linear(MODEL["HIDDEN_DIM"] // 2, 2),  # Width and height
        )

        # Define data augmentation transforms
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=AUGMENTATION["RANDOM_HORIZONTAL_FLIP_PROB"]),
            transforms.RandomRotation(AUGMENTATION["RANDOM_ROTATION_DEGREES"]),
            transforms.ColorJitter(
                brightness=AUGMENTATION["COLOR_JITTER_BRIGHTNESS"], 
                contrast=AUGMENTATION["COLOR_JITTER_CONTRAST"]
            ),
            transforms.RandomAffine(
                degrees=0, 
                translate=(AUGMENTATION["RANDOM_AFFINE_TRANSLATE"], AUGMENTATION["RANDOM_AFFINE_TRANSLATE"])
            ),
        ])

        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=TRAINING["BATCH_SIZE"],
            shuffle=True,
            num_workers=TRAINING["NUM_WORKERS"],
            pin_memory=True,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=TRAINING["BATCH_SIZE"],
            num_workers=TRAINING["NUM_WORKERS"],
            pin_memory=True,
        )

        # Move models to device
        self.cnn = self.cnn.to(self.device)
        self.corner_lstm = self.corner_lstm.to(self.device)
        self.dimension_predictor = self.dimension_predictor.to(self.device)

        # Initialize optimizers
        self.cnn_optimizer = optim.Adam(
            self.cnn.parameters(),
            lr=TRAINING["LEARNING_RATE"],
            weight_decay=TRAINING["WEIGHT_DECAY"],
        )
        self.lstm_optimizer = optim.Adam(
            self.corner_lstm.parameters(),
            lr=TRAINING["LEARNING_RATE"],
            weight_decay=TRAINING["WEIGHT_DECAY"],
        )
        self.dim_optimizer = optim.Adam(
            self.dimension_predictor.parameters(),
            lr=TRAINING["LEARNING_RATE"],
            weight_decay=TRAINING["WEIGHT_DECAY"],
        )

        # Initialize learning rate schedulers
        self.cnn_scheduler = ReduceLROnPlateau(
            self.cnn_optimizer, 
            mode='max', 
            factor=TRAINING["LR_SCHEDULER_FACTOR"], 
            patience=TRAINING["LR_SCHEDULER_PATIENCE"], 
            verbose=True
        )
        self.lstm_scheduler = ReduceLROnPlateau(
            self.lstm_optimizer, 
            mode='max', 
            factor=TRAINING["LR_SCHEDULER_FACTOR"], 
            patience=TRAINING["LR_SCHEDULER_PATIENCE"], 
            verbose=True
        )
        self.dim_scheduler = ReduceLROnPlateau(
            self.dim_optimizer, 
            mode='min', 
            factor=TRAINING["LR_SCHEDULER_FACTOR"], 
            patience=TRAINING["LR_SCHEDULER_PATIENCE"], 
            verbose=True
        )

        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.corner_loss = nn.BCEWithLogitsLoss()
        self.dimension_loss = nn.MSELoss()

        # Training tracking
        self.best_accuracy = 0
        self.best_model_state = None
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def check_and_train(self, model_path: str, num_epochs: int = TRAINING["DEFAULT_EPOCHS"]):
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
        """Train all components of the module"""
        if self.dataset_loader is None:
            raise ValueError(
                "Cannot train without dataset. Please initialize with dataset_path."
            )

        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
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

                # Apply data augmentation
                if hasattr(self, 'train_transforms'):
                    # Note: This is a simplified approach. In a real implementation,
                    # you would need to modify the dataset class to apply transforms
                    # or use a custom collate function
                    pass

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

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.cnn.parameters(), TRAINING["GRADIENT_CLIP_NORM"]
                )
                torch.nn.utils.clip_grad_norm_(
                    self.corner_lstm.parameters(), TRAINING["GRADIENT_CLIP_NORM"]
                )
                torch.nn.utils.clip_grad_norm_(
                    self.dimension_predictor.parameters(), TRAINING["GRADIENT_CLIP_NORM"]
                )

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
                    print(
                        f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {total_batch_loss.item():.4f}"
                    )

            # Calculate training metrics
            train_loss = total_loss / len(self.train_loader)
            train_acc = correct_predictions / total_samples
            
            # Validation phase
            val_loss, val_accuracy = self.validate()
            
            # Update learning rate schedulers
            if self.cnn_scheduler:
                self.cnn_scheduler.step(val_accuracy)
            if self.lstm_scheduler:
                self.lstm_scheduler.step(val_accuracy)
            if self.dim_scheduler:
                self.dim_scheduler.step(val_loss)
                
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_accuracy)
            
            # Save checkpoint
            if epoch % TRAINING["CHECKPOINT_FREQUENCY"] == 0:
                self._save_checkpoint(epoch, val_accuracy, val_loss)
            
            # Early stopping check
            if val_accuracy > self.best_accuracy + self.early_stopping_min_delta:
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
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                
            # Check if early stopping should be triggered
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            epoch_time = time.time() - epoch_start_time
            print(
                f"Epoch {epoch} completed in {epoch_time:.2f}s. "
                f"Training Loss: {train_loss:.4f}, "
                f"Training Accuracy: {100.*train_acc:.2f}%, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {100.*val_accuracy:.2f}%"
            )
            
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")

    def _save_checkpoint(self, epoch, val_accuracy, val_loss):
        """Save a checkpoint during training"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
        )
        checkpoint = {
            "epoch": epoch,
            "cnn": self.cnn.state_dict(),
            "lstm": self.corner_lstm.state_dict(),
            "dim_predictor": (
                self.dimension_predictor.state_dict()
                if self.dimension_predictor
                else None
            ),
            "cnn_optimizer": self.cnn_optimizer.state_dict(),
            "lstm_optimizer": self.lstm_optimizer.state_dict(),
            "dim_optimizer": self.dim_optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "training_history": self.training_history,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def validate(self) -> Tuple[float, float]:
        """Validate the model and return validation loss and accuracy"""
        self.cnn.eval()
        self.corner_lstm.eval()
        self.dimension_predictor.eval()
        
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.valid_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                dimensions = batch["dimensions"].to(self.device)

                # Forward pass
                class_output, features = self.cnn(images)
                dim_output = self.dimension_predictor(features)
                
                # Calculate losses
                class_loss = self.classification_loss(class_output, labels)
                dim_loss = self.dimension_loss(dim_output, dimensions)
                batch_loss = class_loss + dim_loss
                
                val_loss += batch_loss.item()
                
                # Calculate accuracy
                _, predicted = class_output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(self.valid_loader)
        val_accuracy = correct / total
        
        return val_loss, val_accuracy

    def extract_dimensions(self, image: np.ndarray, corners=None) -> torch.Tensor:
        """Extract dimensions from pattern image"""
        if corners is not None and len(corners) > 0:
            try:
                # Ensure corners is a proper numpy array
                corners_array = np.array(corners).reshape(-1, 2)
                
                # Check if we have enough points
                if corners_array.shape[0] < MODEL["MIN_CORNER_POINTS"]:
                    raise ValueError("Not enough corner points to calculate dimensions")
                
                # Find extreme points
                top = corners_array[corners_array[:, 1].argmin()]
                bottom = corners_array[corners_array[:, 1].argmax()]
                left = corners_array[corners_array[:, 0].argmin()]
                right = corners_array[corners_array[:, 0].argmax()]

                # Calculate distances
                height = np.sqrt((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2)
                width = np.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)
                
                # Check for valid dimensions
                if height <= 0 or width <= 0:
                    raise ValueError("Invalid dimensions calculated from corners")

                return torch.tensor([width, height], dtype=torch.float32)
            except Exception as e:
                print(f"Error extracting dimensions from corners: {e}")
                # Fall through to the fallback method

        # Fallback: Use image dimensions with aspect ratio preserved
        h, w = image.shape[:2]
        scale = min(IMAGE_PROCESSING["STANDARD_IMAGE_SIZE"] / w, IMAGE_PROCESSING["STANDARD_IMAGE_SIZE"] / h)
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
            processed_image = preprocess_image_for_model(
                pattern_image_copy, self.device
            )

            with torch.no_grad():
                # Get features
                class_output, features = self.cnn(processed_image)

                # Get pattern type if trained with dataset
                if hasattr(self, "train_dataset") and hasattr(
                    self.train_dataset, "pattern_types"
                ):
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
                    dimensions = (
                        self.dimension_predictor(features).cpu().numpy().squeeze()
                    )
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
            edges = cv2.Canny(
                gray,
                IMAGE_PROCESSING["EDGE_DETECTION_LOW"],
                IMAGE_PROCESSING["EDGE_DETECTION_HIGH"],
            )
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Create a minimal contour if none found
            if len(contours) == 0:
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
