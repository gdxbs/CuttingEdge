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

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    """Environment for pattern packing optimization

    A reinforcement learning environment that simulates pattern placement on cloth.
    Handles the physics and constraints of proper pattern placement.

    Attributes:
        cloth_data: Data about the cloth material
        patterns: List of patterns to place
        rotation_angles: Possible rotation angles
        cloth_space: 2D array representing the cloth state
        placed_patterns: List of already placed patterns
    """

    def __init__(
        self, cloth_data: Dict, patterns: List[Dict], rotation_angles: List[int]
    ):
        """Initialize the packing environment

        Args:
            cloth_data: Dictionary containing cloth properties and dimensions
            patterns: List of pattern dictionaries with contours and dimensions
            rotation_angles: List of angles (in degrees) to try for pattern placement
        """
        # Store input parameters
        self.cloth_data = cloth_data
        self.cloth_width, self.cloth_height = map(int, cloth_data["dimensions"])
        self.patterns = patterns
        self.rotation_angles = rotation_angles

        # Initialize state variables
        self.cloth_space = None  # Will be initialized in reset()
        self.placed_patterns: List[Dict] = []
        self.available_patterns = list(range(len(patterns)))

        # Create cloth boundary as Shapely polygon
        # First try to use the cloth_polygon if provided in cloth_data
        if "cloth_polygon" in cloth_data and cloth_data["cloth_polygon"] is not None:
            self.cloth_polygon = cloth_data["cloth_polygon"]
            # Make sure the polygon is valid
            if not self.cloth_polygon.is_valid:
                self.cloth_polygon = self.cloth_polygon.buffer(
                    0
                )  # Fix potential issues
                if not self.cloth_polygon.is_valid:
                    # Fall back to rectangular boundary
                    self.cloth_polygon = sg.box(
                        0, 0, self.cloth_width, self.cloth_height
                    )
        else:
            # Default to rectangular boundary
            self.cloth_polygon = sg.box(0, 0, self.cloth_width, self.cloth_height)

        # Process patterns into Shapely polygons for collision detection
        self._process_patterns()

        # Define action space: combined discrete action
        # We'll use a MultiDiscrete space:
        # - First dimension selects pattern (n_patterns options)
        # - Second dimension selects x position (width options)
        # - Third dimension selects y position (height options)
        # - Fourth dimension selects rotation (n_rotations options)
        self.action_space = gym.spaces.MultiDiscrete(
            [
                len(patterns),  # Pattern selection
                self.cloth_width,  # X position
                self.cloth_height,  # Y position
                len(rotation_angles),  # Rotation
            ]
        )

        # Observation space: cloth state + available patterns indicator
        # Shape: [1 (cloth occupancy) + len(patterns) (pattern availability)] x height x width
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1 + len(patterns), self.cloth_height, self.cloth_width),
            dtype=np.float32,
        )

    def _process_patterns(self) -> None:
        """Convert patterns to Shapely polygons for collision detection"""
        self.pattern_polygons = []
        self.pattern_idx_map = []

        # Process each pattern
        for pattern_idx, pattern in enumerate(self.patterns):
            if "contours" in pattern and pattern["contours"]:
                # Process each contour in the pattern
                for contour in pattern["contours"]:
                    try:
                        # Skip very small contours
                        contour_points = contour.squeeze()
                        if len(contour_points) < 4:
                            continue

                        # Ensure contour is closed
                        if not np.array_equal(contour_points[0], contour_points[-1]):
                            contour_points = np.vstack(
                                [contour_points, contour_points[0]]
                            )

                        # Create and validate Shapely polygon
                        polygon = sg.Polygon(contour_points)
                        if not polygon.is_valid:
                            continue

                        # Store polygon and its original pattern index
                        self.pattern_polygons.append(polygon)
                        self.pattern_idx_map.append(pattern_idx)
                    except Exception as e:
                        logger.warning(f"Error creating polygon from contour: {e}")
            else:
                # Create rectangle from dimensions if contours not available
                self._create_polygon_from_dimensions(pattern, pattern_idx)

        # Create default polygon if none were created
        if not self.pattern_polygons:
            logger.warning("No valid patterns found, using fallback rectangle")
            polygon = sg.box(0, 0, 10.0, 10.0)
            self.pattern_polygons.append(polygon)
            self.pattern_idx_map.append(0)

    def _create_polygon_from_dimensions(self, pattern: Dict, pattern_idx: int) -> None:
        """Create a rectangular polygon from dimensions

        Args:
            pattern: The pattern data dictionary
            pattern_idx: The index of the pattern
        """
        if "dimensions" in pattern:
            width, height = pattern["dimensions"]
            if hasattr(width, "item"):  # Handle tensor types
                width, height = width.item(), height.item()
            polygon = sg.box(0, 0, float(width), float(height))
            self.pattern_polygons.append(polygon)
            self.pattern_idx_map.append(pattern_idx)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Initial observation and empty info dictionary
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Initialize empty cloth space
        self.cloth_space = np.zeros(
            (self.cloth_height, self.cloth_width), dtype=np.uint8
        )

        # Reset pattern tracking
        self.placed_patterns = []
        self.available_patterns = list(range(len(self.patterns)))

        # Create initial state
        observation = self._create_observation()

        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step (place one pattern)

        Args:
            action: Array with [pattern_idx, x_pos, y_pos, rotation_idx]

        Returns:
            observation: Updated environment state
            reward: Reward for the action
            terminated: Whether episode is complete
            truncated: Whether episode was truncated
            info: Additional info dict
        """
        # Unpack the action
        pattern_idx, x_pos, y_pos, rotation_idx = action

        # Get rotation angle
        rotation = (
            self.rotation_angles[rotation_idx]
            if rotation_idx < len(self.rotation_angles)
            else 0
        )

        # Debug info
        logger.debug(
            f"Action: pattern={pattern_idx}, pos=({x_pos},{y_pos}), rotation={rotation}"
        )
        logger.debug(f"Available patterns: {self.available_patterns}")
        logger.debug(
            f"Pattern map length: {len(self.pattern_idx_map) if hasattr(self, 'pattern_idx_map') else 'N/A'}"
        )
        logger.debug(
            f"Patterns length: {len(self.patterns) if hasattr(self, 'patterns') else 'N/A'}"
        )

        # Invalid action if pattern already used
        if pattern_idx not in self.available_patterns:
            reward = -1.0  # Penalty
            done = len(self.placed_patterns) == len(self.patterns)
            info = {
                "success": False,
                "utilization": self._calculate_utilization(),
                "remaining_patterns": len(self.patterns) - len(self.placed_patterns),
                "message": "Pattern already used",
            }
            logger.debug(
                f"Pattern {pattern_idx} already used. Available: {self.available_patterns}"
            )
            return self._create_observation(), reward, done, False, info

        # Get the original pattern with bounds checking
        try:
            if pattern_idx < len(self.pattern_idx_map):
                original_pattern_idx = self.pattern_idx_map[pattern_idx]
            else:
                original_pattern_idx = (
                    pattern_idx if pattern_idx < len(self.patterns) else 0
                )

            if original_pattern_idx >= len(self.patterns):
                logger.warning(f"Invalid original pattern index {original_pattern_idx}")
                original_pattern_idx = 0

            pattern = self.patterns[original_pattern_idx]
            logger.debug(
                f"Selected pattern {original_pattern_idx} (from pattern_idx {pattern_idx})"
            )

        except Exception as e:
            logger.error(f"Error getting pattern: {e}")
            # Use first pattern as fallback
            pattern = (
                self.patterns[0] if len(self.patterns) > 0 else {"dimensions": (10, 10)}
            )
            logger.warning("Using fallback pattern")

        # For supervised testing during initial development, try placing in center
        if len(self.placed_patterns) == 0 and np.random.random() < 0.1:
            # Try a placement in the center with fallback
            center_x, center_y = self.cloth_width // 2, self.cloth_height // 2
            success = self._place_pattern(pattern, (center_x, center_y), 0)
            if success:
                logger.info(f"Placed first pattern at center: ({center_x}, {center_y})")
                self._track_placed_pattern(
                    pattern, (center_x, center_y), 0, pattern_idx
                )
                reward = 1.0

                # Create info dictionary
                info = {
                    "success": True,
                    "utilization": self._calculate_utilization(),
                    "remaining_patterns": len(self.patterns)
                    - len(self.placed_patterns),
                    "message": "Aided placement at center",
                }
                return self._create_observation(), reward, False, False, info

        # Try to place the pattern with the agent's action
        success = self._place_pattern(pattern, (x_pos, y_pos), rotation, pattern_idx)

        # Debug placement result
        if success:
            logger.info(
                f"Successfully placed pattern {pattern_idx} at ({x_pos}, {y_pos}) with rotation {rotation}"
            )
        else:
            logger.debug(
                f"Failed to place pattern {pattern_idx} at ({x_pos}, {y_pos}) with rotation {rotation}"
            )

        # Calculate reward
        if success:
            reward = self._calculate_reward()
            self._track_placed_pattern(pattern, (x_pos, y_pos), rotation, pattern_idx)
        else:
            reward = -0.5  # Smaller penalty for invalid placement

            # For debugging during training, occasionally try a random valid placement
            if np.random.random() < 0.05 and len(self.patterns) > 0:
                # Try a few random positions
                for _ in range(5):
                    rand_x = np.random.randint(0, self.cloth_width)
                    rand_y = np.random.randint(0, self.cloth_height)
                    rand_rot = np.random.choice(self.rotation_angles)

                    if self._place_pattern(
                        pattern, (rand_x, rand_y), rand_rot, pattern_idx
                    ):
                        logger.info(
                            f"Assisted placement of pattern {pattern_idx} at ({rand_x}, {rand_y})"
                        )
                        self._track_placed_pattern(
                            pattern, (rand_x, rand_y), rand_rot, pattern_idx
                        )
                        reward = 0.1  # Small positive reward
                        success = True
                        break

        # Check if episode is done
        done = len(self.placed_patterns) == len(self.patterns)

        # Prevent very long episodes
        max_steps = len(self.patterns) * 5
        truncated = len(self.placed_patterns) > max_steps

        # Create info dictionary
        info = {
            "success": success,
            "utilization": self._calculate_utilization(),
            "remaining_patterns": len(self.patterns) - len(self.placed_patterns),
        }

        return self._create_observation(), reward, done, truncated, info

    def _track_placed_pattern(
        self, pattern: Dict, position: Tuple[int, int], rotation: int, pattern_idx: int
    ) -> None:
        """Track a successfully placed pattern

        Args:
            pattern: The pattern data
            position: (x, y) position of placement
            rotation: Rotation angle in degrees
            pattern_idx: Index of the pattern
        """
        self.placed_patterns.append(
            {
                "pattern": pattern,
                "position": position,
                "rotation": rotation,
                "pattern_idx": pattern_idx,
            }
        )
        self.available_patterns.remove(pattern_idx)

    def _place_pattern(
        self,
        pattern: Dict,
        position: Tuple[int, int],
        rotation: int,
        pattern_polygon_idx: Optional[int] = None,
    ) -> bool:
        """Place a pattern at the given position and rotation

        Args:
            pattern: Pattern data dictionary
            position: (x, y) position tuple
            rotation: Rotation angle in degrees
            pattern_polygon_idx: Index of the pattern polygon to use

        Returns:
            Whether the placement was successful
        """
        x, y = position

        # Ensure position is within cloth bounds (basic sanity check)
        if x < 0 or x >= self.cloth_width or y < 0 or y >= self.cloth_height:
            logger.debug(
                f"Position ({x}, {y}) outside cloth bounds ({self.cloth_width}x{self.cloth_height})"
            )
            return False

        # Get the pattern polygon
        polygon_idx = self._get_polygon_index(pattern, pattern_polygon_idx)

        # Return failure if no polygon found
        if polygon_idx == -1 or polygon_idx >= len(self.pattern_polygons):
            logger.warning(f"Invalid polygon index: {polygon_idx}")

            # For debugging - during training, sometimes accept placements anyway
            # to help the model learn
            if len(self.placed_patterns) == 0 and np.random.random() < 0.1:
                # Create a small default polygon
                width = min(50.0, self.cloth_width * 0.1)
                height = min(50.0, self.cloth_height * 0.1)
                simple_rect = sg.box(0, 0, width, height)

                # Ensure position is well within bounds
                safe_x = min(max(width, x), self.cloth_width - width * 2)
                safe_y = min(max(height, y), self.cloth_height - height * 2)

                simple_placed = sa.translate(simple_rect, safe_x, safe_y)

                # If this rectangular pattern would be valid, place it
                if self.cloth_polygon.contains(
                    simple_placed
                ) and not self._check_overlap(simple_placed):
                    self._update_cloth_space(simple_placed)
                    logger.info(f"Placed fallback pattern at ({safe_x}, {safe_y})")
                    return True

            return False

        try:
            # Get the pattern polygon and transform it
            pattern_polygon = self.pattern_polygons[polygon_idx]

            # Get pattern bounds before rotation to understand its size
            minx, miny, maxx, maxy = pattern_polygon.bounds
            width, height = maxx - minx, maxy - miny

            # Rotate the pattern
            rotated_polygon = sa.rotate(pattern_polygon, rotation, origin=(0, 0))

            # Find bounds after rotation
            rot_minx, rot_miny, rot_maxx, rot_maxy = rotated_polygon.bounds
            rot_width, rot_height = rot_maxx - rot_minx, rot_maxy - rot_miny

            # Adjust position to ensure it stays within bounds
            adj_x = min(max(0, x), self.cloth_width - rot_width)
            adj_y = min(max(0, y), self.cloth_height - rot_height)

            # Place with adjusted position
            placed_polygon = sa.translate(rotated_polygon, adj_x, adj_y)

            # Do a solid containment check for the polygon

            # Find the intersection of the placed pattern with the cloth
            try:
                intersection = placed_polygon.intersection(self.cloth_polygon)
                intersection_area = intersection.area
                pattern_area = placed_polygon.area

                # If less than 95% of the pattern is within the cloth, reject it
                if intersection_area < 0.95 * pattern_area:
                    logger.debug(
                        f"Pattern not sufficiently contained within cloth: {intersection_area/pattern_area:.2%}"
                    )
                    return False
            except Exception as e:
                logger.warning(f"Error calculating intersection: {e}")
                # If there's an error with the geometry calculations, fall back to simpler check
                if not self.cloth_polygon.contains(placed_polygon):
                    return False

            # Check overlap with already placed patterns
            if self._check_overlap(placed_polygon):
                return False

            # Update the cloth space for visualization
            self._update_cloth_space(placed_polygon)

            return True

        except Exception as e:
            logger.error(f"Error placing pattern: {e}")
            return False

    def _get_polygon_index(
        self, pattern: Dict, pattern_polygon_idx: Optional[int] = None
    ) -> int:
        """Get the index of the polygon for a pattern

        Args:
            pattern: The pattern data
            pattern_polygon_idx: Optional explicit polygon index

        Returns:
            Index of the polygon to use
        """
        # If a valid index is provided, use it directly
        if pattern_polygon_idx is not None and pattern_polygon_idx < len(
            self.pattern_polygons
        ):
            return pattern_polygon_idx

        try:
            # Try to find matching polygon by reference equality
            polygon_idx = -1
            for i, p in enumerate(self.patterns):
                if p is pattern:
                    for j, mapped_idx in enumerate(self.pattern_idx_map):
                        if mapped_idx == i:
                            polygon_idx = j
                            break
                    if polygon_idx >= 0:
                        break

            # If not found by reference, try to find by matching image path
            if polygon_idx == -1 and "image_path" in pattern:
                target_path = pattern["image_path"]
                for i, p in enumerate(self.patterns):
                    if p.get("image_path") == target_path:
                        for j, mapped_idx in enumerate(self.pattern_idx_map):
                            if mapped_idx == i:
                                polygon_idx = j
                                break
                        if polygon_idx >= 0:
                            break

            # If no polygon is found, use the first one for this pattern
            if polygon_idx == -1 and len(self.pattern_polygons) > 0:
                # Check if we're in the initial patterns
                for i, pat in enumerate(self.patterns):
                    if pat == pattern or pat.get("image_path") == pattern.get(
                        "image_path"
                    ):
                        # Find first mapped index for this pattern
                        for j, mapped_idx in enumerate(self.pattern_idx_map):
                            if mapped_idx == i:
                                polygon_idx = j
                                break
                        break

                # If still not found, use pattern index if within bounds
                if polygon_idx == -1:
                    for i, p in enumerate(self.patterns):
                        if p is pattern and i < len(self.pattern_polygons):
                            polygon_idx = i
                            break

            # Last resort - just use first available polygon
            if polygon_idx == -1 and len(self.pattern_polygons) > 0:
                logger.warning("Using first available polygon as fallback")
                polygon_idx = 0

            return polygon_idx

        except Exception as e:
            logger.error(f"Error finding polygon index: {e}")
            # Fallback to first polygon if available
            return 0 if len(self.pattern_polygons) > 0 else -1

    def _check_overlap(self, new_polygon: sg.Polygon) -> bool:
        """Check if a new polygon overlaps with existing patterns

        Args:
            new_polygon: The polygon to check

        Returns:
            Whether there is an overlap
        """
        # First check if polygon is inside cloth bounds
        if not self.cloth_polygon.contains(new_polygon):
            return True

        # Add a small buffer to patterns to prevent patterns from being too close
        # Using a small buffer helps to provide spacing between patterns
        try:
            buffered_new_polygon = new_polygon.buffer(2.0)
        except Exception:
            # If buffering fails, use original polygon
            buffered_new_polygon = new_polygon

        # Check against each placed pattern
        for placed in self.placed_patterns:
            try:
                placed_pattern_idx = placed["pattern_idx"]
                placed_pos = placed["position"]
                placed_rot = placed["rotation"]

                if placed_pattern_idx < len(self.pattern_polygons):
                    # Get original polygon and transform it
                    existing_poly = self.pattern_polygons[placed_pattern_idx]
                    existing_poly = sa.rotate(existing_poly, placed_rot, origin=(0, 0))
                    existing_poly = sa.translate(
                        existing_poly, placed_pos[0], placed_pos[1]
                    )

                    # Check for intersection with buffer for spacing
                    if buffered_new_polygon.intersects(existing_poly):
                        return True
            except Exception as e:
                logger.warning(f"Error checking pattern overlap: {e}")
                # If we hit an error, be conservative and say there's an overlap
                return True

        # Also check if placing this would create a geometry with too many disconnected parts
        # This promotes more cohesive, continuous layouts
        try:
            # Create a union of all placed patterns
            all_patterns = None
            for placed in self.placed_patterns:
                placed_pattern_idx = placed["pattern_idx"]
                placed_pos = placed["position"]
                placed_rot = placed["rotation"]

                if placed_pattern_idx < len(self.pattern_polygons):
                    poly = self.pattern_polygons[placed_pattern_idx]
                    poly = sa.rotate(poly, placed_rot, origin=(0, 0))
                    poly = sa.translate(poly, placed_pos[0], placed_pos[1])

                    if all_patterns is None:
                        all_patterns = poly
                    else:
                        all_patterns = all_patterns.union(poly)

            # If we have a valid union, check if adding the new polygon creates too many parts
            if all_patterns is not None:
                union_with_new = all_patterns.union(new_polygon)

                # Count disconnected components in both unions
                old_parts = (
                    1
                    if isinstance(all_patterns, sg.Polygon)
                    else len(all_patterns.geoms)
                )
                new_parts = (
                    1
                    if isinstance(union_with_new, sg.Polygon)
                    else len(union_with_new.geoms)
                )

                # If adding this pattern creates more disconnected parts, discourage it
                # But still allow it for the first few patterns
                if new_parts > old_parts and len(self.placed_patterns) > 2:
                    # Only apply this restriction once we have at least 3 patterns placed
                    return True
        except Exception:
            # If geometry operations fail, just continue
            pass

        return False

    def _update_cloth_space(self, placed_polygon: sg.Polygon) -> None:
        """Update the cloth space with the placed pattern

        Args:
            placed_polygon: The polygon that was placed
        """
        pattern_mask = np.zeros_like(self.cloth_space)

        # Convert Shapely polygon to OpenCV contour
        polygon_points = np.array(placed_polygon.exterior.coords).astype(int)[:-1]
        cv2.fillPoly(pattern_mask, [polygon_points], (1,))

        # Place pattern on cloth space
        self.cloth_space = np.logical_or(self.cloth_space, pattern_mask).astype(
            np.uint8
        )

    def _calculate_reward(self) -> float:
        """Calculate reward based on pattern placement

        Returns:
            Calculated reward value
        """
        utilization = self._calculate_utilization()
        compactness = self._calculate_compactness()
        valid_placement = 1.0

        # Weighted reward
        reward = 0.7 * utilization + 0.2 * compactness + 0.1 * valid_placement

        return reward

    def _calculate_utilization(self) -> float:
        """Calculate material utilization percentage

        Returns:
            Ratio of used area to total area
        """
        used_area: float = float(np.sum(self.cloth_space))
        total_area = self.cloth_width * self.cloth_height
        return used_area / total_area

    def _calculate_compactness(self) -> float:
        """Calculate how compactly patterns are placed

        Returns:
            Compactness score
        """
        if np.sum(self.cloth_space) == 0:
            return 0.0

        # Find the bounding box of the used area
        rows = np.any(self.cloth_space, axis=1)
        cols = np.any(self.cloth_space, axis=0)

        height: float = float(np.sum(rows))
        width: float = float(np.sum(cols))

        used_area: float = float(np.sum(self.cloth_space))
        bounding_area = height * width

        return used_area / bounding_area if bounding_area > 0 else 0.0

    def _create_observation(self) -> np.ndarray:
        """Create the observation state

        Returns:
            Observation array with cloth state and pattern availability
        """
        # The observation will have channels:
        # [0] - Current cloth state
        # [1:1+n_patterns] - One-hot encoding of available patterns

        # Start with cloth space as first channel
        cloth_space = (
            self.cloth_space
            if self.cloth_space is not None
            else np.zeros((self.cloth_height, self.cloth_width), dtype=np.uint8)
        )
        observation = [cloth_space.copy().astype(np.float32)]

        # Add available patterns channels (one-hot encoding)
        for i in range(len(self.patterns)):
            pattern_channel = np.zeros(
                (self.cloth_height, self.cloth_width), dtype=np.float32
            )
            if i in self.available_patterns:
                # Draw the pattern shape in its channel if available
                if i < len(self.patterns):
                    pattern = self.patterns[i]
                    if "contours" in pattern and pattern["contours"]:
                        try:
                            cv2.fillPoly(pattern_channel, pattern["contours"], (1.0,))
                        except Exception:
                            # Use a simple representation if drawing fails
                            pattern_channel.fill(1.0)
                    else:
                        pattern_channel.fill(1.0)
            observation.append(pattern_channel)

        # Stack channels
        return cast(np.ndarray, np.stack(observation))

    def render(self) -> Any:
        """Render the current state

        Returns:
            RGB image of the current state
        """
        # Create RGB visualization
        rgb_image = np.zeros((self.cloth_height, self.cloth_width, 3), dtype=np.uint8)

        # Draw background
        rgb_image.fill(240)  # Light gray background

        # Draw placed patterns with different colors
        for i, placed in enumerate(self.placed_patterns):
            # Pattern mask not needed in render - removed
            pattern_idx = placed["pattern_idx"]

            # Get the pattern polygon
            if pattern_idx < len(self.pattern_polygons):
                polygon = self.pattern_polygons[pattern_idx]

                # Transform the polygon
                placed_rot = placed["rotation"]
                placed_pos = placed["position"]

                rotated = sa.rotate(polygon, placed_rot, origin=(0, 0))
                translated = sa.translate(rotated, placed_pos[0], placed_pos[1])

                # Draw the polygon
                try:
                    points = np.array(translated.exterior.coords).astype(np.int32)[:-1]

                    # Use a different color for each pattern (simplified color wheel)
                    color_idx = i % 6
                    colors = [
                        (255, 0, 0),  # Red
                        (0, 255, 0),  # Green
                        (0, 0, 255),  # Blue
                        (255, 255, 0),  # Yellow
                        (255, 0, 255),  # Magenta
                        (0, 255, 255),  # Cyan
                    ]

                    # Draw filled polygon
                    cv2.fillPoly(rgb_image, [points], colors[color_idx])

                    # Draw boundary
                    cv2.polylines(rgb_image, [points], True, (50, 50, 50), 1)
                except Exception as e:
                    logger.warning(f"Error rendering pattern: {e}")

        return cast(np.ndarray, rgb_image)


class ClothCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for the cloth packing environment.

    This extracts features from the cloth state and available patterns
    in a way that's useful for making pattern placing decisions.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
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
            n_flatten = 4096
            logger.warning(f"Using default flattened size: {n_flatten}")

        # Final linear layers to get features_dim output
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
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
        self.rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]

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
        """Process cloth mask to extract contours and dimensions

        Args:
            cloth_data: Cloth data dictionary to be modified
        """
        cloth_mask = cloth_data.get("cloth_mask", None)
        if cloth_mask is not None and np.sum(cloth_mask) > 0:
            try:
                contours, _ = cv2.findContours(
                    cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours and len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 500:
                        cloth_data["contours"] = [largest_contour]

                        # Update dimensions
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        if w > 10 and h > 10:
                            cloth_data["dimensions"] = np.array(
                                [w, h], dtype=np.float32
                            )
            except Exception as e:
                logger.warning(f"Error processing cloth mask: {e}")

    def _ensure_valid_cloth_dimensions(self, cloth_data: Dict) -> None:
        """Ensure cloth has valid dimensions

        Args:
            cloth_data: Cloth data dictionary to be modified
        """
        dimensions = cloth_data.get("dimensions", None)
        if dimensions is None or np.any(np.array(dimensions) < 10):
            dimensions = np.array([512.0, 512.0], dtype=np.float32)
            cloth_data["dimensions"] = dimensions

    def _scale_cloth_if_needed(self, cloth_data: Dict) -> None:
        """Scale down cloth if it's too large

        Args:
            cloth_data: Cloth data dictionary to be modified
        """
        cloth_width, cloth_height = cloth_data["dimensions"]
        max_size = 512

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

                # Scale patterns if cloth was scaled
                if need_scaling:
                    self._scale_pattern_contours(processed_pattern, scale)

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

        # Create PPO policy with custom CNN - with simplified architecture for debugging
        policy_kwargs = {
            "features_extractor_class": ClothCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            # Use a simpler network architecture for now
            "net_arch": [64, {"pi": [32], "vf": [32]}],
        }

        # Create model with error handling
        try:
            logger.info("Creating PPO model with custom CNN policy")
            # Check environment observation and action spaces
            logger.info(f"Environment observation space: {env.observation_space}")
            logger.info(f"Environment action space: {env.action_space}")

            # Create the model
            model = PPO(
                "CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=None,  # Disable tensorboard
                device=self.device,
            )
            logger.info("PPO model created successfully")
        except Exception as e:
            logger.error(f"Error creating PPO model: {e}")
            logger.info("Attempting to use default policy without custom CNN")
            # Try with default policy as fallback
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=None,  # Disable tensorboard
                device=self.device,
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
            # For debugging, use a very small number of timesteps first
            debug_timesteps = min(500, actual_timesteps)
            logger.info(f"Initial debug training for {debug_timesteps} timesteps")

            model.learn(
                total_timesteps=debug_timesteps,
                callback=list(callbacks),
                progress_bar=False,  # Disable progress bar
            )

            # If small training worked, continue with full training if needed
            if debug_timesteps < actual_timesteps:
                logger.info(
                    f"Continuing training for remaining {actual_timesteps - debug_timesteps} timesteps"
                )
                model.learn(
                    total_timesteps=actual_timesteps - debug_timesteps,
                    callback=list(callbacks),
                    progress_bar=False,  # Disable progress bar
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

            best_state, best_utilization = self._evaluate_model(model, eval_env)
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Create a default result
            best_state = np.zeros((512, 512), dtype=np.uint8)
            best_utilization = 0.0

        return {"best_state": best_state, "best_utilization": best_utilization}

    def _evaluate_model(
        self, model: PPO, env: PackingEnvironment, n_eval_episodes: int = 5
    ) -> Tuple[Optional[np.ndarray], float]:
        """Evaluate the trained model

        Args:
            model: The trained model
            env: The environment to evaluate on
            n_eval_episodes: Number of episodes to evaluate

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
                    obs, _ = env.reset()
                    done = False
                    step_count = 0
                    max_steps = 50  # Limit steps for debugging

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
                                    env.cloth_space.copy()
                                    if env.cloth_space is not None
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
            cloth_width, cloth_height = env.cloth_width, env.cloth_height
            best_state = np.zeros((cloth_height, cloth_width), dtype=np.uint8)

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

        # Load or create model
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                try:
                    # Load pretrained model
                    policy_kwargs = {
                        "features_extractor_class": ClothCNN,
                        "features_extractor_kwargs": {"features_dim": 256},
                    }

                    self.model = cast(
                        PPO,
                        PPO.load(
                            self.model_path,
                            env=env,
                            device=self.device,
                            custom_objects={"policy_kwargs": policy_kwargs},
                        ),
                    )
                    logger.info(f"Loaded model from {self.model_path}")
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    # Create a simple model
                    self.model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=None)
                    logger.info("Created fallback model after loading error")
            else:
                # Train a quick model if no saved model is available
                logger.warning(
                    "No pretrained model available, training for 5000 timesteps"
                )
                return self.train(cloth_data, patterns_data, num_timesteps=5000)

        # Run inference with fallback
        try:
            logger.info("Running pattern fitting inference")
            obs, _ = env.reset()
            done = False

            # Track best state
            best_utilization = 0.0
            best_state = None

            # Limit steps to avoid infinite loops
            max_steps = len(processed_patterns) * 10
            step = 0
            attempts_without_success = 0
            max_failures = 10  # After this many failures, try manual placement

            while not done and step < max_steps:
                try:
                    # Get action from model
                    action, _ = cast(PPO, self.model).predict(obs, deterministic=True)

                    # Take step in environment
                    obs, _, terminated, truncated, info = env.step(action)

                    # Track best state
                    utilization = info["utilization"]

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

                            # Try different positions with a grid-based approach
                            placed = False

                            # Calculate pattern size to avoid overlaps and boundary issues
                            pattern_width, pattern_height = (
                                50,
                                50,
                            )  # Default fallback size
                            if "dimensions" in pattern:
                                try:
                                    dims = pattern["dimensions"]
                                    if hasattr(dims, "tolist"):
                                        dims = dims.tolist()
                                    pattern_width, pattern_height = map(float, dims)

                                    # Scale if tensor values are too small (normalized values)
                                    if pattern_width < 1.0 and pattern_height < 1.0:
                                        pattern_width *= env.cloth_width * 0.2
                                        pattern_height *= env.cloth_height * 0.2
                                except Exception:
                                    logger.warning(
                                        "Failed to get pattern dimensions, using defaults"
                                    )

                            # Create a grid of positions to try
                            grid_size = (
                                10  # Number of positions to try in each dimension
                            )
                            # Add margin to avoid overlaps (not used directly)

                            # Define a grid of positions that stays well within cloth boundaries
                            min_x = pattern_width * 0.6
                            max_x = env.cloth_width - pattern_width * 1.2
                            min_y = pattern_height * 0.6
                            max_y = env.cloth_height - pattern_height * 1.2

                            # Ensure we have valid bounds
                            if min_x >= max_x or min_y >= max_y:
                                min_x, min_y = 0, 0
                                max_x = env.cloth_width * 0.8
                                max_y = env.cloth_height * 0.8

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

                            # Only try the first 20 positions to avoid excessive computation
                            positions = positions[:20]

                            # Add the center position as a priority
                            center_x = env.cloth_width // 2 - pattern_width // 2
                            center_y = env.cloth_height // 2 - pattern_height // 2
                            positions.insert(0, (int(center_x), int(center_y)))

                            # Try each position with different rotations
                            for pos in positions:
                                for rot in [0, 90, 180, 270]:
                                    if env._place_pattern(
                                        pattern, pos, rot, pattern_idx
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
                                obs = env._create_observation()
                                break

                        # Reset counter even if no placement was successful
                        attempts_without_success = 0

                    if utilization > best_utilization:
                        best_utilization = utilization
                        best_state = (
                            env.cloth_space.copy()
                            if env.cloth_space is not None
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
            for placement in env.placed_patterns:
                placements.append(
                    {
                        "pattern_id": placement["pattern_idx"],
                        "position": placement["position"],
                        "rotation": placement["rotation"],
                    }
                )

            # If no placements were made, try advanced manual placement
            if not placements:
                logger.warning(
                    "No patterns were placed. Trying advanced manual placement."
                )

                # Create a simple environment for placement testing
                test_env = PackingEnvironment(
                    cloth_data=processed_cloth,
                    patterns=processed_patterns,
                    rotation_angles=self.rotation_angles,
                )

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
                                pattern_width *= test_env.cloth_width * 0.2
                                pattern_height *= test_env.cloth_height * 0.2
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

                # Grid-based placement
                cloth_width, cloth_height = test_env.cloth_width, test_env.cloth_height

                # Start placing from the center
                center_x, center_y = cloth_width // 2, cloth_height // 2

                # Try to place each pattern
                for idx, pattern, _area, width, height in sorted_patterns:
                    placed = False

                    # Generate a spiral of positions to try, starting from center
                    positions = []
                    max_radius = min(cloth_width, cloth_height) // 2

                    # Add the center position first
                    positions.append((center_x, center_y))

                    # Add positions in increasing distance from center
                    for radius in range(max_radius // 10, max_radius, max_radius // 10):
                        for angle in range(
                            0, 360, 30
                        ):  # Try positions every 30 degrees
                            x = center_x + int(radius * np.cos(np.radians(angle)))
                            y = center_y + int(radius * np.sin(np.radians(angle)))
                            positions.append((x, y))

                    # Try all positions with different rotations
                    for rot in [0, 90, 180, 270]:
                        for pos_x, pos_y in positions:
                            # Adjust position to center the pattern
                            adj_x = max(
                                0,
                                min(cloth_width - int(width), pos_x - int(width) // 2),
                            )
                            adj_y = max(
                                0,
                                min(
                                    cloth_height - int(height), pos_y - int(height) // 2
                                ),
                            )

                            if test_env._place_pattern(
                                pattern, (adj_x, adj_y), rot, idx
                            ):
                                test_env._track_placed_pattern(
                                    pattern, (adj_x, adj_y), rot, idx
                                )
                                placements.append(
                                    {
                                        "pattern_id": idx,
                                        "position": (adj_x, adj_y),
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
                        cloth_bounds = self.cloth_polygon.bounds
                        min_x, min_y, max_x, max_y = cloth_bounds

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
                                        width *= cloth_width * 0.2
                                        height *= cloth_height * 0.2
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
                                if self.cloth_polygon.contains(test_poly):
                                    # We found a valid position
                                    placements.append(
                                        {
                                            "pattern_id": idx,
                                            "position": (int(adj_x), int(adj_y)),
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
                                if self.cloth_polygon.contains(test_rect):
                                    placements.append(
                                        {
                                            "pattern_id": idx,
                                            "position": (pos_x, pos_y),
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
            final_state = best_state if best_state is not None else env.cloth_space
            final_utilization = (
                best_utilization
                if best_utilization > 0
                else env._calculate_utilization()
            )

            logger.info(
                f"Pattern fitting complete. Final utilization: {final_utilization:.4f}"
            )

            return {
                "final_state": final_state,
                "utilization": final_utilization,
                "placements": placements,
                "cloth_dims": (env.cloth_width, env.cloth_height),
                "patterns": processed_patterns,
            }

        except Exception as e:
            logger.error(f"Error during pattern fitting: {e}")

            # Create fallback result
            return {
                "final_state": np.zeros((512, 512), dtype=np.uint8),
                "utilization": 0.0,
                "placements": [],
                "cloth_dims": (512, 512),
                "patterns": processed_patterns,
            }
