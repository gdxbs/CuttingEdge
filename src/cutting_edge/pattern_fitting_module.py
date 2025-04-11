import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import cv2
import gymnasium as gym
import numpy as np
import shapely.affinity as sa
import shapely.geometry as sg
import torch
import torch.nn as nn

# Import Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from cutting_edge.config import (
    IMAGE_PROCESSING,
    MODEL,
    PATTERN_FITTING,
    TRAINING,
    VISUALIZATION,
    ENVIRONMENT,
)

# Configure logging
logging.basicConfig(level=getattr(logging, ENVIRONMENT["LOG_LEVEL"]))
logger = logging.getLogger(__name__)

"""
Pattern Fitting Module

This module implements pattern fitting onto cloth materials using reinforcement learning.
The system optimizes material utilization by determining:
1. Which pattern to place next
2. Where and at what rotation to place it

This implementation uses Stable Baselines3 for the reinforcement learning component,
making it more maintainable and easier to understand.
"""


class PackingEnvironment(gym.Env):
    """Environment for pattern fitting using reinforcement learning"""
    # These attributes need to be defined at class level to prevent attribute errors
    cloth_width = PATTERN_FITTING["DEFAULT_CLOTH_WIDTH"]
    cloth_height = PATTERN_FITTING["DEFAULT_CLOTH_HEIGHT"]

    def __init__(
        self,
        cloth_data: Dict,
        patterns: List[Dict],
        rotation_angles: List[float] = PATTERN_FITTING["ROTATION_ANGLES"],
    ):
        """Initialize the environment

        Args:
            cloth_data: Dictionary with cloth properties
            patterns: List of pattern dictionaries
            rotation_angles: List of possible rotation angles
        """
        super().__init__()

        # Store cloth and pattern data
        self.cloth_data = cloth_data
        self.patterns = patterns
        self.rotation_angles = rotation_angles

        # Initialize cloth state
        self.cloth_state = np.zeros(
            (cloth_data["width"], cloth_data["height"]), dtype=np.uint8
        )
        self.current_pattern_idx = 0
        self.placed_patterns = []

        # Define action space
        # [x, y, rotation_idx]
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array(
                [
                    cloth_data["width"] - 1,
                    cloth_data["height"] - 1,
                    len(rotation_angles) - 1,
                ]
            ),
            dtype=np.int32,
        )

        # Define observation space
        # [cloth_state, current_pattern_mask, remaining_patterns_mask]
        self.observation_space = gym.spaces.Dict(
            {
                "cloth_state": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(cloth_data["width"], cloth_data["height"], 1),
                    dtype=np.float32,
                ),
                "current_pattern": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(cloth_data["width"], cloth_data["height"], 1),
                    dtype=np.float32,
                ),
                "pattern_info": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(4,),  # [width, height, area, remaining_patterns]
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, seed=None):
        """Reset the environment

        Returns:
            Initial observation
        """
        super().reset(seed=seed)
        self.cloth_state = np.zeros(
            (self.cloth_width, self.cloth_height), dtype=np.uint8
        )
        self.current_pattern_idx = 0
        self.placed_patterns = []
        self.available_patterns = list(range(len(self.patterns)))
        self.pattern_idx_map = {i: i for i in range(len(self.patterns))}
        return self._get_observation(), {}

    def step(self, action):
        """Take a step in the environment

        Args:
            action: [x, y, rotation_idx]

        Returns:
            observation, reward, terminated, truncated, info
        """
        x, y, rotation_idx = action
        pattern = self.patterns[self.current_pattern_idx]
        rotation = self.rotation_angles[rotation_idx]

        # Try to place pattern
        success = self._place_pattern(pattern, x, y, rotation, self.current_pattern_idx)

        # Calculate reward components
        placement_reward = 1.0 if success else -1.0
        utilization_reward = self._calculate_utilization()
        overlap_penalty = -0.5 if not success else 0.0
        efficiency_bonus = 0.1 if self._is_efficient_placement(x, y, pattern, rotation) else 0.0

        # Combine rewards
        reward = (
            placement_reward +
            utilization_reward * 0.5 +
            overlap_penalty +
            efficiency_bonus
        )

        # Update state
        if success:
            # Track pattern placement
            self._track_placed_pattern(pattern, (x, y), rotation, self.current_pattern_idx)
            self.current_pattern_idx += 1

        # Check if episode is done
        done = (
            self.current_pattern_idx >= len(self.patterns)
            # Comment out for now, can uncomment after fixing _has_valid_placements
            # or not self._has_valid_placements()
        )

        # Create info dictionary with additional metrics
        info = {
            "success": success,
            "utilization": utilization_reward,
            "current_pattern": self.current_pattern_idx,
            "total_patterns": len(self.patterns),
            "placed_count": len(self.placed_patterns) if hasattr(self, "placed_patterns") else 0
        }

        return self._get_observation(), reward, done, False, info

    def _place_pattern(self, pattern: Dict, x: int, y: int, rotation: float, pattern_idx: int = 0) -> bool:
        """Try to place a pattern on the cloth

        Args:
            pattern: Pattern dictionary
            x: X coordinate
            y: Y coordinate
            rotation: Rotation angle
            pattern_idx: Optional pattern index for visualization

        Returns:
            True if placement was successful
        """
        # Get pattern dimensions after rotation
        width, height = self._get_rotated_dimensions(pattern, rotation)

        # Check if pattern fits within cloth boundaries
        if (
            x + width > self.cloth_data["width"]
            or y + height > self.cloth_data["height"]
        ):
            return False

        # Create pattern mask for this position
        pattern_mask = np.zeros_like(self.cloth_state)
        pattern_mask[y:y + height, x:x + width] = 1

        # Check for overlaps with existing patterns
        if np.any(self.cloth_state & pattern_mask):
            return False

        # Check if pattern is fully contained within the cloth mask (if available)
        if 'cloth_mask' in self.cloth_data and self.cloth_data['cloth_mask'] is not None:
            cloth_mask = self.cloth_data['cloth_mask']
            # Resize mask if dimensions don't match
            if (cloth_mask.shape[0] != self.cloth_state.shape[0] or 
                cloth_mask.shape[1] != self.cloth_state.shape[1]):
                try:
                    cloth_mask = cv2.resize(
                        cloth_mask, 
                        (self.cloth_state.shape[1], self.cloth_state.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                except Exception as e:
                    # If resize fails, skip the cloth mask check
                    pass
            else:
                # Check if pattern is at least MIN_PATTERN_COVERAGE within the cloth mask
                pattern_area = np.sum(pattern_mask)
                overlap_with_cloth = np.sum(pattern_mask & (cloth_mask > 0))
                
                # Only place if at least MIN_PATTERN_COVERAGE of pattern is within cloth
                if overlap_with_cloth < pattern_area * PATTERN_FITTING["MIN_PATTERN_COVERAGE"]:
                    return False

        # Place pattern
        if pattern_idx > 0:
            # Use pattern index + 1 for visualization (with different ID for each pattern)
            self.cloth_state[y:y + height, x:x + width] = pattern_idx + 1
        else:
            # Use 1 for standard placement (binary mask)
            self.cloth_state[y:y + height, x:x + width] = 1
        
        return True
        
    def _track_placed_pattern(self, pattern: Dict, pos: Tuple[int, int], rotation: float, pattern_idx: int) -> None:
        """Track a placed pattern

        Args:
            pattern: Pattern dictionary
            pos: (x, y) coordinates
            rotation: Rotation angle
            pattern_idx: Pattern index
        """
        if not hasattr(self, 'placed_patterns'):
            self.placed_patterns = []
            
        # Add to placed patterns list
        self.placed_patterns.append({
            "pattern": pattern,
            "x": pos[0],
            "y": pos[1],
            "rotation": rotation,
            "pattern_idx": pattern_idx
        })

    def _get_rotated_dimensions(self, pattern: Dict, rotation: float) -> Tuple[int, int]:
        """Get pattern dimensions after rotation

        Args:
            pattern: Pattern dictionary
            rotation: Rotation angle

        Returns:
            Width and height after rotation
        """
        width = pattern["width"]
        height = pattern["height"]

        if rotation in [90, 270]:
            return height, width
        return width, height

    def _calculate_utilization(self) -> float:
        """Calculate cloth utilization

        Returns:
            Utilization ratio
        """
        # Get total cloth area (using cloth mask if available)
        if hasattr(self, 'cloth_data') and 'cloth_mask' in self.cloth_data and self.cloth_data['cloth_mask'] is not None:
            cloth_mask = self.cloth_data['cloth_mask']
            # Resize mask if dimensions don't match
            if (cloth_mask.shape[0] != self.cloth_height or 
                cloth_mask.shape[1] != self.cloth_width):
                try:
                    cloth_mask = cv2.resize(
                        cloth_mask, 
                        (self.cloth_width, self.cloth_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                except Exception as e:
                    # Fallback to default calculation
                    total_area = self.cloth_width * self.cloth_height
            
            total_area = np.sum(cloth_mask > 0)
            if total_area == 0:  # Safety check
                total_area = self.cloth_width * self.cloth_height
        else:
            total_area = self.cloth_width * self.cloth_height
            
        # Calculate used area - only count pixels that are inside the cloth mask
        if hasattr(self, 'cloth_data') and 'cloth_mask' in self.cloth_data and self.cloth_data['cloth_mask'] is not None:
            cloth_mask = self.cloth_data['cloth_mask']
            # Resize mask if dimensions don't match
            if (cloth_mask.shape[0] != self.cloth_height or 
                cloth_mask.shape[1] != self.cloth_width):
                try:
                    cloth_mask = cv2.resize(
                        cloth_mask, 
                        (self.cloth_width, self.cloth_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                except Exception:
                    # If resize fails, just use the full state
                    used_area = np.sum(self.cloth_state)
                    return used_area / total_area
                    
            # Only count area within the cloth mask
            used_area = np.sum(self.cloth_state & (cloth_mask > 0))
        else:
            used_area = np.sum(self.cloth_state)
            
        # Return the ratio (with safety check)
        if total_area > 0:
            return used_area / total_area
        return 0.0

    def _is_efficient_placement(self, x: int, y: int, pattern: Dict, rotation: float) -> bool:
        """Check if pattern placement is efficient

        Args:
            x: X coordinate
            y: Y coordinate
            pattern: Pattern dictionary
            rotation: Rotation angle

        Returns:
            True if placement is efficient
        """
        # Check if pattern is placed next to another pattern or cloth edge
        width, height = self._get_rotated_dimensions(pattern, rotation)
        
        # Check left edge
        if x > 0 and np.any(self.cloth_state[y:y + height, x - 1]):
            return True
            
        # Check right edge
        if x + width < self.cloth_width and np.any(self.cloth_state[y:y + height, x + width]):
            return True
            
        # Check top edge
        if y > 0 and np.any(self.cloth_state[y - 1, x:x + width]):
            return True
            
        # Check bottom edge
        if y + height < self.cloth_height and np.any(self.cloth_state[y + height, x:x + width]):
            return True
            
        return False

    def _has_valid_placements(self) -> bool:
        """Check if there are any valid placements remaining

        Returns:
            True if valid placements exist
        """
        if self.current_pattern_idx >= len(self.patterns):
            return False

        pattern = self.patterns[self.current_pattern_idx]
        
        # Try each possible rotation
        for rotation in self.rotation_angles:
            width, height = self._get_rotated_dimensions(pattern, rotation)
            
            # Check each possible position
            for y in range(self.cloth_height - height + 1):
                for x in range(self.cloth_width - width + 1):
                    # Check if position is valid
                    pattern_mask = np.zeros_like(self.cloth_state)
                    pattern_mask[y:y + height, x:x + width] = 1
                    
                    if not np.any(self.cloth_state & pattern_mask):
                        return True
                        
        return False

    def _get_observation(self) -> Dict:
        """Get current observation

        Returns:
            Observation dictionary
        """
        # Get current pattern
        if self.current_pattern_idx < len(self.patterns):
            current_pattern = self.patterns[self.current_pattern_idx]
        else:
            current_pattern = self.patterns[-1]  # Use last pattern as placeholder

        # Create pattern mask
        pattern_mask = np.zeros_like(self.cloth_state)
        # Get height and width ensuring they are integers
        try:
            p_height = int(current_pattern["height"])
            p_width = int(current_pattern["width"])
            # Ensure they are within bounds and at least 1
            p_height = max(1, min(p_height, self.cloth_height))
            p_width = max(1, min(p_width, self.cloth_width))
            pattern_mask[:p_height, :p_width] = 1
        except (ValueError, TypeError, KeyError):
            # Fallback for any issues
            pattern_mask[:1, :1] = 1

        # Calculate pattern info
        remaining_patterns = len(self.patterns) - self.current_pattern_idx
        try:
            p_width = float(current_pattern.get("width", 50))
            p_height = float(current_pattern.get("height", 50))
            pattern_info = np.array(
                [
                    p_width / self.cloth_width,
                    p_height / self.cloth_height,
                    (p_width * p_height) / (self.cloth_width * self.cloth_height),
                    remaining_patterns / len(self.patterns),
                ],
                dtype=np.float32,
            )
        except (ValueError, TypeError):
            # Default values if conversion fails
            pattern_info = np.array([0.1, 0.1, 0.01, 
                          remaining_patterns / len(self.patterns)], 
                          dtype=np.float32)

        return {
            "cloth_state": self.cloth_state[..., np.newaxis].astype(np.float32),
            "current_pattern": pattern_mask[..., np.newaxis].astype(np.float32),
            "pattern_info": pattern_info,
        }


class ClothCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for the cloth packing environment.

    This extracts features from the cloth state and available patterns
    in a way that's useful for making pattern placing decisions.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = MODEL["FEATURE_DIM"]):
        """
        Initialize the feature extractor.

        Args:
            observation_space: The environment's observation space
            features_dim: Output feature dimension
        """
        super(ClothCNN, self).__init__(observation_space, features_dim)

        # Input channels from observation space
        n_input_channels = observation_space.shape[0]
        logger.info(f"ClothCNN initializing with {n_input_channels} input channels")
        logger.info(f"Observation space shape: {observation_space.shape}")

        # Simple CNN architecture
        self.cnn = nn.Sequential(
            # First convolutional block
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Third convolutional block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Global pooling
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
        )

        # Compute output size of CNN - with error handling
        try:
            with torch.no_grad():
                sample = torch.zeros((1,) + observation_space.shape).float()
                logger.info(f"Created sample tensor of shape {sample.shape}")
                n_flatten = self.cnn(sample).shape[1]
                logger.info(f"Flattened features shape: {n_flatten}")
        except Exception as e:
            logger.error(f"Error during CNN output size computation: {e}")
            # Use a reasonable default
            n_flatten = MODEL["DEFAULT_FLATTENED_SIZE"]
            logger.warning(f"Using default flattened size: {n_flatten}")

        # Final linear layers to get features_dim output
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, MODEL["HIDDEN_DIM"]),
            nn.ReLU(),
            nn.Linear(MODEL["HIDDEN_DIM"], features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN

        Args:
            observations: Input observation tensor

        Returns:
            Extracted features
        """
        features = self.cnn(observations)
        return cast(torch.Tensor, self.linear(features))


class PatternFittingModule:
    """Module for fitting patterns onto cloth

    Main interface for pattern fitting functionality using reinforcement learning.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the pattern fitting module

        Args:
            model_path: Path to saved model or None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model: Optional[PPO] = None
        self.rotation_angles = PATTERN_FITTING["ROTATION_ANGLES"]

        logger.info(f"Pattern Fitting Module initialized on device: {self.device}")

    def prepare_data(
        self, cloth_data: Dict, patterns_data: List[Dict]
    ) -> Tuple[Dict, List[Dict]]:
        """Prepare cloth and patterns data for the environment

        Args:
            cloth_data: Dictionary with cloth properties
            patterns_data: List of pattern dictionaries

        Returns:
            Tuple of (processed_cloth, processed_patterns)
        """
        # Process cloth data
        processed_cloth = self._process_cloth_data(cloth_data)

        # Process patterns data
        processed_patterns = self._process_patterns_data(patterns_data, processed_cloth)

        logger.info(f"Prepared {len(processed_patterns)} patterns for fitting")
        return processed_cloth, processed_patterns

    def _process_cloth_data(self, cloth_data: Dict) -> Dict:
        """Process cloth data for the environment

        Args:
            cloth_data: Raw cloth data dictionary

        Returns:
            Processed cloth data dictionary
        """
        processed_cloth = cloth_data.copy()

        # Use cloth mask if available
        self._process_cloth_mask(processed_cloth)

        # Ensure valid dimensions
        self._ensure_valid_cloth_dimensions(processed_cloth)

        # Scale down very large cloth
        self._scale_cloth_if_needed(processed_cloth)

        return processed_cloth

    def _process_cloth_mask(self, cloth_data: Dict) -> None:
        """Process cloth mask to extract contours and dimensions"""
        cloth_mask = cloth_data.get("cloth_mask", None)
        if cloth_mask is not None and np.sum(cloth_mask) > 0:
            try:
                # Find the largest contour directly
                contours, _ = cv2.findContours(
                    cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Get the largest contour by area
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    # Only process if area is significant
                    if area > IMAGE_PROCESSING["MIN_DIMENSION"]:
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Store the contour and dimensions
                        cloth_data["contours"] = [largest_contour]
                        cloth_data["dimensions"] = np.array([w, h], dtype=np.float32)
                        cloth_data["area"] = area
                        
                        # Create a simplified polygon for pattern fitting
                        cloth_data["cloth_polygon"] = sg.Polygon(largest_contour.squeeze())
                        
            except Exception as e:
                logger.warning(f"Error processing cloth mask: {e}")

    def _ensure_valid_cloth_dimensions(self, cloth_data: Dict) -> None:
        """Ensure cloth has valid dimensions

        Args:
            cloth_data: Cloth data dictionary to be modified
        """
        dimensions = cloth_data.get("dimensions", None)
        if dimensions is None or np.any(np.array(dimensions) < IMAGE_PROCESSING["MIN_DIMENSION"]):
            dimensions = np.array([IMAGE_PROCESSING["STANDARD_IMAGE_SIZE"], IMAGE_PROCESSING["STANDARD_IMAGE_SIZE"]], dtype=np.float32)
            cloth_data["dimensions"] = dimensions

    def _scale_cloth_if_needed(self, cloth_data: Dict) -> None:
        """Scale down cloth if it's too large

        Args:
            cloth_data: Cloth data dictionary to be modified
        """
        cloth_width, cloth_height = cloth_data["dimensions"]
        max_size = IMAGE_PROCESSING["MAX_CLOTH_SIZE"]

        if cloth_width > max_size or cloth_height > max_size:
            scale = max_size / max(cloth_width, cloth_height)
            cloth_data["dimensions"] = (cloth_width * scale, cloth_height * scale)

            # Scale contours if they exist
            if "contours" in cloth_data and cloth_data["contours"]:
                try:
                    for i, contour in enumerate(cloth_data["contours"]):
                        cloth_data["contours"][i] = contour * scale
                except Exception as e:
                    logger.warning(f"Error scaling contours: {e}")

    def _process_patterns_data(
        self, patterns_data: List[Dict], processed_cloth: Dict
    ) -> List[Dict]:
        """Process patterns data for the environment

        Args:
            patterns_data: List of raw pattern dictionaries
            processed_cloth: Processed cloth data dictionary

        Returns:
            List of processed pattern dictionaries
        """
        processed_patterns = []
        cloth_width, cloth_height = processed_cloth["dimensions"]
        max_size = 512
        need_scaling = cloth_width > max_size or cloth_height > max_size
        scale = max_size / max(cloth_width, cloth_height) if need_scaling else 1.0

        for pattern in patterns_data:
            try:
                processed_pattern = pattern.copy()

                # Ensure pattern has valid contours
                if (
                    "contours" not in processed_pattern
                    or not processed_pattern["contours"]
                ):
                    self._create_contours_from_dimensions(processed_pattern)

                # Add width and height to the pattern directly
                if "dimensions" in processed_pattern:
                    dims = processed_pattern["dimensions"]
                    if hasattr(dims, "tolist"):
                        dims = dims.tolist()
                    pattern_width, pattern_height = map(float, dims)
                    
                    # Scale if values are too small (normalized)
                    if pattern_width < 1.0 and pattern_height < 1.0:
                        pattern_width = int(pattern_width * processed_cloth["width"] * 0.2)
                        pattern_height = int(pattern_height * processed_cloth["height"] * 0.2)
                    
                    processed_pattern["width"] = pattern_width
                    processed_pattern["height"] = pattern_height
                else:
                    # Default sizes if dimensions not available
                    processed_pattern["width"] = 50
                    processed_pattern["height"] = 50

                # Scale patterns if cloth was scaled
                if need_scaling:
                    self._scale_pattern_contours(processed_pattern, scale)
                    processed_pattern["width"] = processed_pattern["width"] * scale
                    processed_pattern["height"] = processed_pattern["height"] * scale

                processed_patterns.append(processed_pattern)
            except Exception as e:
                logger.warning(f"Error processing pattern: {e}")

        return processed_patterns

    def _create_contours_from_dimensions(self, pattern: Dict) -> None:
        """Create rectangle contour from dimensions

        Args:
            pattern: Pattern data dictionary to be modified
        """
        if "dimensions" in pattern:
            # Create rectangle contour from dimensions
            dims = pattern["dimensions"]
            if hasattr(dims, "tolist"):
                dims = dims.tolist()
            width, height = map(float, dims)

            # Create rectangle
            rect = np.array(
                [[[0, 0]], [[width, 0]], [[width, height]], [[0, height]]],
                dtype=np.float32,
            )

            pattern["contours"] = [rect]

    def _scale_pattern_contours(self, pattern: Dict, scale: float) -> None:
        """Scale pattern contours by given factor

        Args:
            pattern: Pattern data dictionary to be modified
            scale: Scale factor to apply
        """
        try:
            for i, contour in enumerate(pattern["contours"]):
                pattern["contours"][i] = contour * scale
        except Exception as e:
            logger.warning(f"Error scaling pattern contours: {e}")

    def train(
        self,
        cloth_data: Dict,
        patterns_data: List[Dict],
        num_episodes: int = 1,
        num_timesteps: int = 50000,
    ) -> Dict:
        """Train the pattern fitting model

        Args:
            cloth_data: Dictionary with cloth properties
            patterns_data: List of pattern dictionaries
            num_episodes: Number of training episodes (for compatibility with main.py)
            num_timesteps: Number of training timesteps

        Returns:
            Training result dictionary
        """
        # Convert episodes to timesteps if provided
        actual_timesteps = num_timesteps
        if num_episodes > 0:
            actual_timesteps = num_episodes * 5000  # Approximate conversion

        # Prepare data
        processed_cloth, processed_patterns = self.prepare_data(
            cloth_data, patterns_data
        )

        # Create environment
        env = PackingEnvironment(
            cloth_data=processed_cloth,
            patterns=processed_patterns,
            rotation_angles=self.rotation_angles,
        )

        # Create PPO policy with custom CNN and improved architecture
        policy_kwargs = {
            "features_extractor_class": ClothCNN,
            "features_extractor_kwargs": {"features_dim": MODEL["FEATURE_DIM"]},
            "net_arch": [
                dict(
                    pi=[MODEL["HIDDEN_DIM"], MODEL["HIDDEN_DIM"], MODEL["HIDDEN_DIM"] // 2],  # Policy network
                    vf=[MODEL["HIDDEN_DIM"], MODEL["HIDDEN_DIM"], MODEL["HIDDEN_DIM"] // 2],  # Value network
                )
            ],
            "activation_fn": nn.ReLU,
            "ortho_init": True,
        }

        # Create model with error handling
        try:
            logger.info("Creating PPO model with custom CNN policy")
            # Check environment observation and action spaces
            logger.info(f"Environment observation space: {env.observation_space}")
            logger.info(f"Environment action space: {env.action_space}")

            # Create the model with improved hyperparameters
            model = PPO(
                "MultiInputPolicy",  # Use MultiInputPolicy for Dict observation space
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=TRAINING["LEARNING_RATE"],
                n_steps=PATTERN_FITTING["MAX_STEPS"] * 40,  # Scale up for better learning
                batch_size=TRAINING["BATCH_SIZE"],
                n_epochs=TRAINING["DEFAULT_EPOCHS"],
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=ENVIRONMENT["TENSORBOARD_LOG_DIR"],
                device=torch.device(ENVIRONMENT["DEVICE"] if torch.cuda.is_available() else "cpu"),
            )
            logger.info("PPO model created successfully")
        except Exception as e:
            logger.error(f"Error creating PPO model: {e}")
            logger.info("Attempting to use default policy without custom CNN")
            # Try with default policy as fallback
            model = PPO(
                "MultiInputPolicy",  # Use MultiInputPolicy for Dict observation space
                env,
                learning_rate=TRAINING["LEARNING_RATE"],
                n_steps=PATTERN_FITTING["MAX_STEPS"] * 40,
                batch_size=TRAINING["BATCH_SIZE"],
                n_epochs=TRAINING["DEFAULT_EPOCHS"],
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=ENVIRONMENT["TENSORBOARD_LOG_DIR"],
                device=torch.device(ENVIRONMENT["DEVICE"] if torch.cuda.is_available() else "cpu"),
            )

        # Training callbacks
        callbacks: Sequence[CheckpointCallback] = [
            CheckpointCallback(
                save_freq=10000,
                save_path="./checkpoints/",
                name_prefix="ppo_pattern_fitting",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )
        ]

        # Train the model with error handling
        try:
            logger.info(f"Starting training for {actual_timesteps} timesteps")
            # Use a larger initial training period
            initial_timesteps = min(5000, actual_timesteps)
            logger.info(f"Initial training for {initial_timesteps} timesteps")

            model.learn(
                total_timesteps=initial_timesteps,
                callback=list(callbacks),
                progress_bar=True,
            )

            # If initial training worked, continue with full training if needed
            if initial_timesteps < actual_timesteps:
                logger.info(
                    f"Continuing training for remaining {actual_timesteps - initial_timesteps} timesteps"
                )
                model.learn(
                    total_timesteps=actual_timesteps - initial_timesteps,
                    callback=list(callbacks),
                    progress_bar=True,
                )

            logger.info("Training completed successfully")

            # Save the model
            if self.model_path:
                try:
                    model.save(self.model_path)
                    logger.info(f"Model saved to {self.model_path}")
                except Exception as e:
                    logger.error(f"Error saving model to {self.model_path}: {e}")

            # Save model reference
            self.model = cast(PPO, model)

        except Exception as e:
            logger.error(f"Error during training: {e}")
            # Create a dummy model for evaluation
            self.model = model

        # Get best state
        try:
            eval_env = PackingEnvironment(
                cloth_data=processed_cloth,
                patterns=processed_patterns,
                rotation_angles=self.rotation_angles,
            )

            best_state, best_utilization = self._evaluate_model(model, eval_env, processed_cloth=processed_cloth)
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Create a default result
            best_state = np.zeros((512, 512), dtype=np.uint8)
            best_utilization = 0.0

        return {"best_state": best_state, "best_utilization": best_utilization}

    def _evaluate_model(
        self, model: PPO, env: PackingEnvironment, n_eval_episodes: int = 5, processed_cloth: Dict = None
    ) -> Tuple[Optional[np.ndarray], float]:
        """Evaluate the trained model

        Args:
            model: The trained model
            env: The environment to evaluate on
            n_eval_episodes: Number of episodes to evaluate
            processed_cloth: The processed cloth data for fallback state creation

        Returns:
            Tuple of (best_state, best_utilization)
        """
        best_utilization = 0.0
        best_state = None

        # Limit the number of episodes for quick debugging
        n_eval_episodes = min(2, n_eval_episodes)

        try:
            for _i in range(n_eval_episodes):
                try:
                    # Manually set up environment attributes before reset
                    if not hasattr(env, 'cloth_width') and hasattr(env, 'cloth_data'):
                        env.cloth_width = int(env.cloth_data.get('width', 512))
                        env.cloth_height = int(env.cloth_data.get('height', 512))
                        
                    obs, _ = env.reset()
                    done = False
                    step_count = 0
                    max_steps = PATTERN_FITTING["MAX_STEPS"]  # Use config value

                    while not done and step_count < max_steps:
                        try:
                            # For prediction errors
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, terminated, truncated, info = env.step(action)
                            done = terminated or truncated
                            step_count += 1

                            # Keep track of best state
                            utilization = info["utilization"]
                            if utilization > best_utilization:
                                best_utilization = utilization
                                best_state = (
                                    env.cloth_state.copy()
                                    if env.cloth_state is not None
                                    else None
                                )

                        except Exception as e:
                            logger.error(f"Error during model prediction or step: {e}")
                            break

                except Exception as e:
                    logger.error(f"Error during episode reset: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")

        # Ensure we have some result
        if best_state is None:
            # Create a default best state
            try:
                # Try to get dimensions from environment first
                if hasattr(env, 'cloth_width') and hasattr(env, 'cloth_height'):
                    cloth_width, cloth_height = env.cloth_width, env.cloth_height
                # Then try from processed_cloth if provided
                elif processed_cloth is not None:
                    cloth_width = processed_cloth.get('width', 512)
                    cloth_height = processed_cloth.get('height', 512)
                # Default as last resort
                else:
                    cloth_width, cloth_height = 512, 512
                
                # Create array with correct dimensions and ensure they're integers
                best_state = np.zeros((int(cloth_height), int(cloth_width)), dtype=np.uint8)
            except Exception as e:
                logger.error(f"Error creating default state: {e}")
                best_state = np.zeros((512, 512), dtype=np.uint8)

        return best_state, best_utilization

    def fit_patterns(
        self, cloth_data: Dict, patterns_data: List[Dict], visualize: bool = False
    ) -> Dict:
        """Fit patterns onto cloth

        Args:
            cloth_data: Dictionary with cloth properties
            patterns_data: List of pattern dictionaries
            visualize: Whether to enable visualization (not used currently)

        Returns:
            Dictionary with fitting results
        """
        # Prepare data
        processed_cloth, processed_patterns = self.prepare_data(
            cloth_data, patterns_data
        )

        # Create environment
        env = PackingEnvironment(
            cloth_data=processed_cloth,
            patterns=processed_patterns,
            rotation_angles=self.rotation_angles,
        )

        # Load model if available
        loaded_model = False
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
                loaded_model = True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Failed to load model, will create a new one")
                
        if not loaded_model:
            try:
                # Create a simple model manually instead of full training
                logger.warning("Creating a simple model for pattern fitting")
                
                # Create a policy
                policy_kwargs = {
                    "net_arch": [
                        dict(
                            pi=[64, 64],  # Simple policy network
                            vf=[64, 64],  # Simple value network
                        )
                    ],
                }
                
                # Create model
                self.model = PPO(
                    "MultiInputPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=TRAINING["LEARNING_RATE"],
                    n_steps=128,
                    batch_size=64,
                    n_epochs=5,
                    gamma=0.99,
                    device=torch.device("cpu"),
                )
                
                # Learn a tiny bit (just a few steps)
                self.model.learn(total_timesteps=100)
                logger.info("Created a simple model for pattern fitting")
            except Exception as e:
                logger.error(f"Error creating simple model: {e}")
                logger.warning("Will proceed with manual pattern placement")

        # Run inference with fallback
        try:
            logger.info("Running pattern fitting inference")
            obs, _ = env.reset()
            done = False

            # Track best state
            best_utilization = 0.0
            best_state = None

            # Limit steps to avoid infinite loops
            max_steps = PATTERN_FITTING["MAX_INFERENCE_STEPS"]
            step = 0
            attempts_without_success = 0
            max_failures = PATTERN_FITTING["MAX_FAILURES_BEFORE_MANUAL"]

            while not done and step < max_steps:
                try:
                    # Ensure environment has required attributes
                    if not hasattr(env, 'cloth_width') and hasattr(env, 'cloth_data'):
                        env.cloth_width = int(env.cloth_data.get('width', 512))
                        env.cloth_height = int(env.cloth_data.get('height', 512))
                        
                    # Get action from model
                    action, _ = cast(PPO, self.model).predict(obs, deterministic=True)

                    # Take step in environment
                    obs, _, terminated, truncated, info = env.step(action)

                    # Track best state with default values if needed
                    utilization = info.get("utilization", 0.0)

                    # Check if placement was successful
                    if info.get("success", False):
                        attempts_without_success = 0
                    else:
                        attempts_without_success += 1

                    # Manual intervention if too many failures
                    if attempts_without_success >= max_failures:
                        logger.warning(
                            f"Too many failed attempts ({attempts_without_success}), trying manual placement"
                        )

                        # Try placing each pattern
                        for pattern_idx in env.available_patterns:
                            # Get the pattern data
                            if pattern_idx >= len(
                                env.pattern_idx_map
                            ) or env.pattern_idx_map[pattern_idx] >= len(env.patterns):
                                continue  # Skip invalid patterns

                            pattern = env.patterns[env.pattern_idx_map[pattern_idx]]

                            # Calculate pattern size to avoid overlaps and boundary issues
                            pattern_width, pattern_height = (
                                IMAGE_PROCESSING["DEFAULT_PATTERN_WIDTH"],
                                IMAGE_PROCESSING["DEFAULT_PATTERN_HEIGHT"],
                            )
                            if "dimensions" in pattern:
                                try:
                                    dims = pattern["dimensions"]
                                    if hasattr(dims, "tolist"):
                                        dims = dims.tolist()
                                    pattern_width, pattern_height = map(float, dims)

                                    # Scale if values are too small (normalized)
                                    if pattern_width < 1.0 and pattern_height < 1.0:
                                        # Use larger scale factor for better cloth utilization
                                        pattern_width *= env.cloth_width * PATTERN_FITTING["PATTERN_SCALE_FACTOR"]
                                        pattern_height *= env.cloth_height * PATTERN_FITTING["PATTERN_SCALE_FACTOR"]
                                        
                                        # Make patterns at least MIN_PATTERN_WIDTH_RATIO of cloth dimensions for visibility
                                        min_width = env.cloth_width * PATTERN_FITTING["MIN_PATTERN_WIDTH_RATIO"]
                                        min_height = env.cloth_height * PATTERN_FITTING["MIN_PATTERN_HEIGHT_RATIO"]
                                        pattern_width = max(pattern_width, min_width)
                                        pattern_height = max(pattern_height, min_height)
                                except Exception:
                                    logger.warning(
                                        "Failed to get pattern dimensions, using defaults"
                                    )

                            # Create a grid of positions to try
                            grid_size = PATTERN_FITTING["GRID_SIZE"]

                            # Define a grid of positions that stays well within cloth boundaries
                            min_x = pattern_width * PATTERN_FITTING["PATTERN_MARGIN_X"]
                            max_x = env.cloth_width - pattern_width * PATTERN_FITTING["PATTERN_MARGIN_X"]
                            min_y = pattern_height * PATTERN_FITTING["PATTERN_MARGIN_Y"]
                            max_y = env.cloth_height - pattern_height * PATTERN_FITTING["PATTERN_MARGIN_Y"]

                            # Ensure we have valid bounds
                            if min_x >= max_x or min_y >= max_y:
                                min_x, min_y = 0, 0
                                max_x = env.cloth_width * PATTERN_FITTING["CLOTH_BOUNDARY_MARGIN"]
                                max_y = env.cloth_height * PATTERN_FITTING["CLOTH_BOUNDARY_MARGIN"]

                            # Generate positions across the cloth
                            positions = []
                            for i in range(grid_size):
                                for j in range(grid_size):
                                    # Create a staggered grid
                                    x = min_x + (max_x - min_x) * (i / grid_size)
                                    y = min_y + (max_y - min_y) * (j / grid_size)
                                    positions.append((int(x), int(y)))

                            # Shuffle positions for variety
                            import random
                            random.shuffle(positions)

                            # Only try the first N positions to avoid excessive computation
                            positions = positions[:PATTERN_FITTING["MAX_POSITIONS_TO_TRY"]]

                            # Add the center position as a priority
                            center_x = env.cloth_width // 2 - pattern_width // 2
                            center_y = env.cloth_height // 2 - pattern_height // 2
                            positions.insert(0, (int(center_x), int(center_y)))

                            # If no positions from cloth mask or very few, add spiral positions
                            if len(positions) < 20:
                                # Start from center of cloth
                                positions.append((center_x, center_y))
                                
                                # Add positions in increasing distance from center
                                for radius in range(max_radius // 10, max_radius, max_radius // 10):
                                    for angle in range(0, 360, PATTERN_FITTING["SPIRAL_ANGLE_STEP"]):  # Try positions every N degrees
                                        x = center_x + int(radius * np.cos(np.radians(angle)))
                                        y = center_y + int(radius * np.sin(np.radians(angle)))
                                        positions.append((x, y))

                            # Try all positions with different rotations
                            for rot in PATTERN_FITTING["ROTATION_ANGLES"]:
                                for pos in positions[:PATTERN_FITTING["MAX_POSITIONS_TO_TRY"]]:
                                    if env._place_pattern(
                                        pattern, pos[0], pos[1], rot, pattern_idx
                                    ):
                                        env._track_placed_pattern(
                                            pattern, pos, rot, pattern_idx
                                        )
                                        logger.info(
                                            f"Manually placed pattern {pattern_idx} at {pos} with rotation {rot}"
                                        )
                                        placed = True
                                        attempts_without_success = 0
                                        break

                                if placed:
                                    break

                            if placed:
                                # Update observation
                                obs = env._get_observation()
                                break

                        # Reset counter even if no placement was successful
                        attempts_without_success = 0

                    if utilization > best_utilization:
                        best_utilization = utilization
                        best_state = (
                            env.cloth_state.copy()
                            if env.cloth_state is not None
                            else None
                        )

                    # Check if done
                    done = terminated or truncated
                    step += 1

                    logger.info(
                        f"Step {step}: Utilization {utilization:.4f}, Success: {info['success']}"
                    )

                except Exception as e:
                    logger.error(f"Error during inference step {step}: {e}")
                    step += 1
                    continue

            # Format results
            placements = []
            if hasattr(env, 'placed_patterns'):
                for placement in env.placed_patterns:
                    placements.append(
                        {
                            "pattern_id": placement.get("pattern_idx", 0),
                            "x": placement.get("x", 0),
                            "y": placement.get("y", 0),
                            "rotation": placement.get("rotation", 0),
                        }
                    )

            # If no placements were made, try advanced manual placement
            if not placements:
                logger.warning(
                    "No patterns were placed. Trying advanced manual placement."
                )
                
                # Initialize info dictionary with success field
                info = {"success": False}

                # Direct manual pattern placement instead of using environment
                # Create simple pattern placements
                cloth_width = int(processed_cloth.get('width', 512))
                cloth_height = int(processed_cloth.get('height', 512))
                best_state = np.zeros((cloth_height, cloth_width), dtype=np.uint8)
                
                # Place pattern pieces as simple rectangles
                pattern_margin = 20  # Space between patterns
                current_x, current_y = 50, 50  # Starting position

                # Place patterns one by one with optimal spacing
                # Track placed patterns in the environment

                # First, sort patterns by area (largest first)
                sorted_patterns = []
                for idx, pattern in enumerate(processed_patterns):
                    # Calculate pattern size
                    pattern_width, pattern_height = 50, 50  # Default
                    if "dimensions" in pattern:
                        try:
                            dims = pattern["dimensions"]
                            if hasattr(dims, "tolist"):
                                dims = dims.tolist()
                            pattern_width, pattern_height = map(float, dims)

                            # Scale if values are too small (normalized)
                            if pattern_width < 1.0 and pattern_height < 1.0:
                                pattern_width *= cloth_width * PATTERN_FITTING["PATTERN_SCALE_FACTOR"]
                                pattern_height *= cloth_height * PATTERN_FITTING["PATTERN_SCALE_FACTOR"]
                                
                                # Make patterns at least MIN_PATTERN_WIDTH_RATIO of cloth dimensions for visibility
                                min_width = cloth_width * PATTERN_FITTING["MIN_PATTERN_WIDTH_RATIO"]
                                min_height = cloth_height * PATTERN_FITTING["MIN_PATTERN_HEIGHT_RATIO"]
                                pattern_width = max(pattern_width, min_width)
                                pattern_height = max(pattern_height, min_height)
                        except Exception:
                            logger.warning(
                                f"Failed to get dimensions for pattern {idx}"
                            )

                    area = pattern_width * pattern_height
                    sorted_patterns.append(
                        (idx, pattern, area, pattern_width, pattern_height)
                    )

                # Sort by area (descending)
                sorted_patterns.sort(key=lambda x: x[2], reverse=True)

                # Grid-based placement - use the already defined cloth_width and cloth_height

                # Start placing from the center
                center_x, center_y = cloth_width // 2, cloth_height // 2

                # Try to place each pattern
                for idx, pattern, _area, width, height in sorted_patterns:
                    placed = False

                    # Generate positions to try, focusing on inside the cloth mask
                    positions = []
                    max_radius = min(cloth_width, cloth_height) // 2
                    
                    # Use cloth mask to generate valid positions if available
                    cloth_mask = processed_cloth.get("cloth_mask")
                    if cloth_mask is not None and cloth_mask.size > 0:
                        # Resize mask if needed to match cloth dimensions
                        if cloth_mask.shape[0] != cloth_height or cloth_mask.shape[1] != cloth_width:
                            try:
                                cloth_mask = cv2.resize(
                                    cloth_mask, 
                                    (cloth_width, cloth_height),
                                    interpolation=cv2.INTER_NEAREST
                                )
                            except Exception as e:
                                logger.warning(f"Error resizing cloth mask for position generation: {e}")
                        
                        # Find non-zero positions (inside the cloth)
                        non_zero_positions = np.argwhere(cloth_mask > 0)
                        # Convert to (x, y) format and sample points throughout the cloth
                        if len(non_zero_positions) > 0:
                            # Take a sample of points spread throughout the cloth
                            # Use a step size that gives us about 100 sample points
                            step = max(1, len(non_zero_positions) // 100)
                            for pos in non_zero_positions[::step]:
                                positions.append((int(pos[1]), int(pos[0])))  # x, y format
                    
                    # If no positions from cloth mask or very few, add spiral positions
                    if len(positions) < 20:
                        # Start from center of cloth
                        positions.append((center_x, center_y))
                        
                        # Add positions in increasing distance from center
                        for radius in range(max_radius // 10, max_radius, max_radius // 10):
                            for angle in range(0, 360, PATTERN_FITTING["SPIRAL_ANGLE_STEP"]):  # Try positions every N degrees
                                x = center_x + int(radius * np.cos(np.radians(angle)))
                                y = center_y + int(radius * np.sin(np.radians(angle)))
                                positions.append((x, y))

                    # Try all positions with different rotations
                    for rot in PATTERN_FITTING["ROTATION_ANGLES"]:
                        for pos in positions[:PATTERN_FITTING["MAX_POSITIONS_TO_TRY"]]:
                            # Adjust position to center the pattern
                            adj_x = max(
                                0,
                                min(cloth_width - int(width), pos[0] - int(width) // 2),
                            )
                            adj_y = max(
                                0,
                                min(
                                    cloth_height - int(height), pos[1] - int(height) // 2
                                ),
                            )

                            # Direct placement without the environment
                            # Create a rectangular mask for the pattern
                            pattern_mask = np.zeros_like(best_state)
                            rot_width = width if rot == 0 else height
                            rot_height = height if rot == 0 else width
                            
                            # Make sure coordinates are valid
                            valid_x = min(max(0, adj_x), cloth_width - int(rot_width))
                            valid_y = min(max(0, adj_y), cloth_height - int(rot_height))
                            
                            # Create mask for this position
                            pattern_mask[valid_y:valid_y+int(rot_height), valid_x:valid_x+int(rot_width)] = 1
                            
                            # Get cloth mask for checking if pattern is inside the cloth
                            cloth_mask = processed_cloth.get("cloth_mask")
                            
                            # Check if placement would overlap existing patterns AND ensure it's inside the cloth
                            if not np.any(best_state & pattern_mask):
                                # Additional check to ensure pattern is inside cloth mask
                                if cloth_mask is not None and cloth_mask.size > 0:
                                    # Resize cloth mask if needed to match best_state dimensions
                                    if cloth_mask.shape != best_state.shape:
                                        try:
                                            cloth_mask = cv2.resize(
                                                cloth_mask, 
                                                (best_state.shape[1], best_state.shape[0]),
                                                interpolation=cv2.INTER_NEAREST
                                            )
                                        except Exception as e:
                                            logger.warning(f"Error resizing cloth mask: {e}")
                                            
                                    # Check if pattern is fully contained within the cloth
                                    pattern_area = np.sum(pattern_mask)
                                    overlap_with_cloth = np.sum(pattern_mask & (cloth_mask > 0))
                                    
                                    # Only place if at least 95% of pattern is within cloth
                                    if overlap_with_cloth >= pattern_area * 0.95:
                                        # Place pattern (with index+1 for visualization)
                                        best_state[valid_y:valid_y+int(rot_height), valid_x:valid_x+int(rot_width)] = idx + 1
                                    else:
                                        # Skip - pattern not fully inside cloth
                                        continue
                                else:
                                    # No cloth mask - just place the pattern
                                    best_state[valid_y:valid_y+int(rot_height), valid_x:valid_x+int(rot_width)] = idx + 1
                                
                                # Add to placements
                                placements.append(
                                    {
                                        "pattern_id": idx,
                                        "x": valid_x,
                                        "y": valid_y,
                                        "rotation": rot,
                                    }
                                )
                                logger.info(
                                    f"Added optimized placement for pattern {idx} at ({adj_x}, {adj_y}) with rotation {rot}"
                                )
                                placed = True
                                break

                        if placed:
                            break

                    # If we couldn't place this pattern, log a warning
                    if not placed:
                        logger.warning(
                            f"Failed to place pattern {idx} in optimized layout"
                        )

                # If still no placements made, fall back to simple placement
                if not placements:
                    logger.warning(
                        "Optimized placement failed. Using simple fallback with cloth bounds check."
                    )

                    # Use a grid placement inside the cloth bounds
                    # First determine the cloth bounds - it might not be a rectangle
                    try:
                        # Use processed_cloth directly instead of self
                        if "cloth_polygon" in processed_cloth and processed_cloth["cloth_polygon"] is not None:
                            cloth_bounds = processed_cloth["cloth_polygon"].bounds
                            min_x, min_y, max_x, max_y = cloth_bounds
                        else:
                            # Fallback to using cloth dimensions
                            min_x, min_y = 0, 0
                            max_x, max_y = processed_cloth["width"], processed_cloth["height"]

                        # Adjust bounds to ensure we're safely inside the cloth
                        safe_min_x = min_x + (max_x - min_x) * 0.1
                        safe_max_x = max_x - (max_x - min_x) * 0.1
                        safe_min_y = min_y + (max_y - min_y) * 0.1
                        safe_max_y = max_y - (max_y - min_y) * 0.1

                        # Create a grid of positions
                        positions = []
                        grid_size = 3  # 3x3 grid

                        for i in range(grid_size):
                            for j in range(grid_size):
                                pos_x = safe_min_x + (safe_max_x - safe_min_x) * (
                                    i / (grid_size - 1)
                                )
                                pos_y = safe_min_y + (safe_max_y - safe_min_y) * (
                                    j / (grid_size - 1)
                                )
                                positions.append((int(pos_x), int(pos_y)))

                        # Use positions for each pattern
                        for idx, pattern in enumerate(processed_patterns):
                            # Calculate pattern size for the pattern
                            width, height = 50, 50  # Default size
                            if "dimensions" in pattern:
                                try:
                                    dims = pattern["dimensions"]
                                    if hasattr(dims, "tolist"):
                                        dims = dims.tolist()
                                    width, height = map(float, dims)

                                    # Scale if values are too small
                                    if width < 1.0 and height < 1.0:
                                        width *= cloth_width * PATTERN_FITTING["PATTERN_SCALE_FACTOR"]
                                        height *= cloth_height * PATTERN_FITTING["PATTERN_SCALE_FACTOR"]
                                        
                                        # Make patterns at least MIN_PATTERN_WIDTH_RATIO of cloth dimensions for visibility
                                        min_width = cloth_width * PATTERN_FITTING["MIN_PATTERN_WIDTH_RATIO"]
                                        min_height = cloth_height * PATTERN_FITTING["MIN_PATTERN_HEIGHT_RATIO"]
                                        width = max(width, min_width)
                                        height = max(height, min_height)
                                except Exception:
                                    logger.warning(
                                        f"Failed to get dimensions for pattern {idx}"
                                    )

                            # Try each position until we find one that works
                            for pos in positions:
                                pos_x, pos_y = pos

                                # Adjust position to fit pattern size
                                adj_x = max(safe_min_x, min(safe_max_x - width, pos_x))
                                adj_y = max(safe_min_y, min(safe_max_y - height, pos_y))

                                # Create a dummy polygon to check placement
                                test_poly = sg.box(
                                    adj_x, adj_y, adj_x + width, adj_y + height
                                )

                                # Check if this position works
                                cloth_polygon = processed_cloth.get("cloth_polygon")
                                if cloth_polygon is not None and cloth_polygon.contains(test_poly):
                                    # Valid position found
                                    # We found a valid position
                                    placements.append(
                                        {
                                            "pattern_id": idx,
                                            "x": int(adj_x),
                                            "y": int(adj_y),
                                            "rotation": 0,
                                        }
                                    )
                                    logger.info(
                                        f"Added validated fallback placement for pattern {idx} at ({int(adj_x)}, {int(adj_y)})"
                                    )
                                    # Remove this position from consideration
                                    positions.remove(pos)
                                    break

                            # Don't add patterns that don't fit - we'll skip them completely
                            if len(placements) <= idx:
                                # Just log the skipped pattern
                                logger.warning(
                                    f"Pattern {idx} could not be placed within cloth boundary - skipping it"
                                )

                    except Exception as e:
                        logger.error(f"Error in fallback placement: {e}")
                        # Very simple placement as last resort but only for patterns that fit
                        for idx, pattern in enumerate(processed_patterns):
                            try:
                                # Create a rectangle for this pattern
                                width, height = 50, 50  # Default size
                                if "dimensions" in pattern:
                                    try:
                                        dims = pattern["dimensions"]
                                        if hasattr(dims, "tolist"):
                                            dims = dims.tolist()
                                        width, height = map(float, dims)
                                        if width < 1.0 and height < 1.0:
                                            width, height = width * 100, height * 100
                                    except Exception:
                                        pass

                                # Create test rectangle
                                pos_x, pos_y = 100, 100 + idx * 100
                                test_rect = sg.box(
                                    pos_x, pos_y, pos_x + width, pos_y + height
                                )

                                # Only add if it fits in the cloth
                                cloth_polygon = processed_cloth.get("cloth_polygon")
                                if cloth_polygon is not None and cloth_polygon.contains(test_rect):
                                    placements.append(
                                        {
                                            "pattern_id": idx,
                                            "x": pos_x,
                                            "y": pos_y,
                                            "rotation": 0,
                                        }
                                    )
                                    logger.info(
                                        f"Added simple emergency fallback for pattern {idx}"
                                    )
                                else:
                                    logger.warning(
                                        f"Skipping pattern {idx} in emergency placement - doesn't fit in cloth"
                                    )
                            except Exception as pattern_err:
                                logger.error(
                                    f"Error placing pattern {idx} in emergency mode: {pattern_err}"
                                )

            # Finalize result
            # Create a fallback final state with proper dimensions if needed
            if best_state is not None:
                final_state = best_state
            elif hasattr(env, 'cloth_state') and env.cloth_state is not None:
                final_state = env.cloth_state
            else:
                # Default empty state
                cloth_width = processed_cloth.get('width', 512)
                cloth_height = processed_cloth.get('height', 512)
                final_state = np.zeros((int(cloth_height), int(cloth_width)), dtype=np.uint8)
                
            # Calculate utilization manually based on the actual placements
            # This is a more accurate method than relying on environment's calculation
            # especially when we're using the manual fallback placement
            try:
                # Get cloth mask for accurate area calculation
                cloth_mask = processed_cloth.get('cloth_mask')
                if cloth_mask is not None and np.sum(cloth_mask) > 0:
                    # Resize mask if dimensions don't match
                    if cloth_mask.shape != final_state.shape:
                        try:
                            cloth_mask = cv2.resize(
                                cloth_mask, 
                                (final_state.shape[1], final_state.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        except Exception as e:
                            logger.warning(f"Error resizing cloth mask for utilization: {e}")
                    
                    # Calculate cloth area (only non-zero pixels)
                    cloth_area = np.sum(cloth_mask > 0)
                    
                    # Calculate area covered by patterns (non-zero in final_state AND inside cloth)
                    pattern_area = np.sum((final_state > 0) & (cloth_mask > 0))
                    
                    # Calculate utilization as pattern area / cloth area
                    if cloth_area > 0:
                        final_utilization = pattern_area / cloth_area
                    else:
                        final_utilization = 0.0
                    
                    logger.info(f"Manual utilization calculation: {pattern_area}/{cloth_area} = {final_utilization:.4f}")
                else:
                    # Fallback to simpler calculation
                    total_area = final_state.shape[0] * final_state.shape[1]
                    used_area = np.sum(final_state > 0)
                    final_utilization = used_area / total_area if total_area > 0 else 0.0
                    logger.info(f"Simple utilization calculation: {used_area}/{total_area} = {final_utilization:.4f}")
            except Exception as e:
                logger.error(f"Error calculating utilization: {e}")
                final_utilization = best_utilization
                if final_utilization <= 0:
                    try:
                        if hasattr(env, '_calculate_utilization'):
                            final_utilization = env._calculate_utilization()
                        else:
                            final_utilization = 0.0
                    except Exception:
                        final_utilization = 0.0

            logger.info(
                f"Pattern fitting complete. Final utilization: {final_utilization:.4f}"
            )

            return {
                "final_state": final_state,
                "utilization": final_utilization,
                "placements": placements,
                "cloth_dims": (processed_cloth.get('width', 512), processed_cloth.get('height', 512)),
                "patterns": processed_patterns,
            }

        except Exception as e:
            logger.error(f"Error during pattern fitting: {e}")

            # Create fallback result with proper dimensions
            width = int(processed_cloth.get('width', 512))
            height = int(processed_cloth.get('height', 512))
            return {
                "final_state": np.zeros((height, width), dtype=np.uint8),
                "utilization": 0.0,
                "placements": [],
                "cloth_dims": (width, height),
                "patterns": processed_patterns,
            }
