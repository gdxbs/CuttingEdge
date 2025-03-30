import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import shapely.affinity as sa  # For rotating and transforming polygons
import shapely.geometry as sg  # For handling complex polygons
import torch
import torch.nn as nn
import gymnasium as gym  # OpenAI Gym for RL environments
import torch.nn.functional as F
import torch.optim as optim

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pattern Fitting Module
#
# This module implements pattern fitting onto cloth materials using Hierarchical
# Reinforcement Learning (HRL). Based on research by Wang et al. (2022) and Zhang et al. (2023),
# the system uses a two-level approach:
#
# 1. A manager network that decides which pattern to place next
# 2. A worker network that decides where and at what rotation to place the pattern
#
# The implementation follows the approach described in the paper:
# "Planning Irregular Object Packing via Hierarchical Reinforcement Learning" (Wang et al., 2022)
#
# The system can optimize material utilization through:
# - Strategic pattern ordering
# - Optimal positioning and orientation
# - Learning-based adaptation to different cloth and pattern shapes


class PackingEnvironment(gym.Env):
    """Environment for pattern packing optimization

    This class implements a reinforcement learning environment for optimizing
    pattern placement on cloth. It handles the simulation of placing pattern
    contours on a 2D cloth space while enforcing constraints such as no overlapping
    and staying within cloth boundaries.

    The environment follows the Gymnasium interface with methods for:
    - reset(): Initializing a new episode
    - step(action): Taking an action and returning next state, reward, done flag

    State representation includes:
    - Current cloth space (binary mask of occupied space)
    - Next pattern to place (binary mask of pattern shape)
    - Distance transform channel (spatial awareness)

    References:
    - "Planning Irregular Object Packing via Hierarchical Reinforcement Learning" (Wang et al., 2022)
    - "Tree Search + Reinforcement Learning for Two-Dimensional Cutting Stock Problem
       With Complex Constraints" (Zhang et al., 2023)
    """

    def __init__(
        self,
        cloth_data: Dict,
        patterns: List[Dict],
        rotation_angles: List[int],
        render_mode=None,
    ):
        """Initialize the packing environment

        Args:
            cloth_data: Dictionary with cloth properties including dimensions
            patterns: List of dictionaries containing pattern properties
            rotation_angles: List of possible rotation angles for patterns
            render_mode: Mode for rendering (human, rgb_array, or None)
        """
        self.cloth_data = cloth_data
        self.cloth_width, self.cloth_height = map(int, cloth_data["dimensions"])
        self.patterns = patterns
        self.current_state = None
        self.placed_patterns = []
        self.available_patterns = list(
            range(len(patterns))
        )  # Initialize available patterns
        self.grid_size = 1  # Minimum unit for placement
        self.rotation_angles = rotation_angles
        self.render_mode = render_mode

        # Create Shapely polygon for cloth boundary
        self.cloth_polygon = sg.box(0, 0, self.cloth_width, self.cloth_height)

        # Convert patterns to Shapely polygons
        self.pattern_polygons = []
        # Keep track of which pattern each polygon belongs to
        self.pattern_idx_map = []
        pattern_count = 0
        
        # Process each pattern and its contours
        for pattern_idx, pattern in enumerate(patterns):
            if "contours" in pattern and pattern["contours"]:
                # Process each contour in the pattern (there might be multiple pattern pieces)
                for contour_idx, contour in enumerate(pattern["contours"]):
                    try:
                        # Reshape contour to the format Shapely expects
                        contour_points = contour.squeeze()
                        
                        # Check if we have enough points for a valid polygon
                        if len(contour_points) < 4:
                            # Skip very small contours or create fallback
                            continue
                        
                        # Make sure it's closed (first and last points match)
                        if not np.array_equal(contour_points[0], contour_points[-1]):
                            contour_points = np.vstack([contour_points, contour_points[0]])
                        
                        # Create a Shapely polygon
                        polygon = sg.Polygon(contour_points)
                        
                        # Verify the polygon is valid
                        if not polygon.is_valid:
                            continue
                            
                        # Add the polygon and map it to the original pattern
                        self.pattern_polygons.append(polygon)
                        self.pattern_idx_map.append(pattern_idx)
                        pattern_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error creating polygon from contour {contour_idx}: {e}")
                
                # If no valid polygons were created, use a fallback
                if len(self.pattern_polygons) == pattern_count:
                    width, height = pattern.get("dimensions", (10, 10))
                    if hasattr(width, "item"):  # Handle tensor types
                        width, height = width.item(), height.item()
                    polygon = sg.box(0, 0, float(width), float(height))
                    self.pattern_polygons.append(polygon)
                    self.pattern_idx_map.append(pattern_idx)
                    pattern_count += 1
            else:
                # Fallback to a simple rectangle if no contours
                if "dimensions" in pattern:
                    width, height = pattern["dimensions"]
                    # Convert to float (handles both tensor and numpy types)
                    if hasattr(width, "item"):
                        width, height = width.item(), height.item()
                    width, height = float(width), float(height)
                    polygon = sg.box(0, 0, width, height)
                    self.pattern_polygons.append(polygon)
                    self.pattern_idx_map.append(pattern_idx)
                    pattern_count += 1
                else:
                    # Log warning for patterns without geometry
                    logger.warning(f"Pattern {pattern_idx} missing both contours and dimensions")
            
        # Ensure we have at least one pattern
        if not self.pattern_polygons:
            # Create a default rectangle as fallback
            logger.warning("No valid patterns found, using fallback rectangle")
            polygon = sg.box(0, 0, 10.0, 10.0)
            self.pattern_polygons.append(polygon)
            self.pattern_idx_map.append(0)

        # Enhanced observation space with more channels for better pattern understanding
        # Add a pattern mask channel for each pattern to help the model understand all
        # available patterns at once, enabling better planning
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(3 + len(patterns), self.cloth_height, self.cloth_width),
            dtype=np.float32,
        )

        # Use Dict action space for compatibility with our HRL implementation
        # Action space: [pattern_idx, position_x, position_y, rotation_idx]
        self.action_space = gym.spaces.Dict(
            {
                "pattern_idx": gym.spaces.Discrete(len(patterns)),
                "position": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.cloth_width - 1, self.cloth_height - 1]),
                    dtype=np.int32,
                ),
                "rotation": gym.spaces.Discrete(len(rotation_angles)),
                # Add finer rotation control
                "fine_rotation": gym.spaces.Box(
                    low=np.array([-5.0]),  # Fine adjustment in degrees
                    high=np.array([5.0]),
                    dtype=np.float32,
                ),
            }
        )

        # Set metadata for rendering
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

        # Initialize cloth space
        self.reset()[0]  # Only keep state, ignore info dict

    def reset(self, seed=None, options=None):
        """Reset environment to initial state

        Returns:
            Tuple of (state, info_dict) following Gymnasium API
        """
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)

        self.cloth_space = np.zeros(
            (self.cloth_height, self.cloth_width), dtype=np.uint8
        )
        self.current_state = self._get_state()
        self.placed_patterns = []
        self.available_patterns = list(
            range(len(self.patterns))
        )  # Indices of available patterns

        return (
            self.current_state,
            {},
        )  # Return state and empty info dict (Gymnasium API)

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment with enhanced placement options
        
        Enhanced to handle:
        1. Fine rotation adjustments for more precise placement
        2. Better rewards for successful placements
        3. Additional tracking of pattern placement quality

        Args:
            action: Dictionary containing pattern_idx, position, rotation,
                   and fine_rotation for precise angular adjustments

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info) following Gymnasium API
        """
        pattern_idx = action["pattern_idx"]
        position = action["position"]

        # Get base rotation from discrete rotation index
        base_rotation = (
            action["rotation"]
            if isinstance(action["rotation"], (int, float))
            else self.rotation_angles[action["rotation"]]
        )

        # Apply fine rotation adjustment if provided
        fine_rotation = action.get("fine_rotation", 0.0)
        if isinstance(fine_rotation, np.ndarray) and len(fine_rotation) > 0:
            fine_rotation = fine_rotation[0]  # Extract value from array

        # Calculate final rotation with fine adjustment
        rotation = base_rotation + float(fine_rotation)

        # Keep rotation in 0-360 range
        rotation = rotation % 360

        # If pattern is already placed
        if pattern_idx not in self.available_patterns:
            reward = -1.0
            done = len(self.placed_patterns) == len(self.patterns)
            info = {
                "success": False,
                "utilization": self._calculate_utilization(),
                "remaining_patterns": len(self.patterns) - len(self.placed_patterns),
                "placement_quality": 0.0,
            }
            return self.current_state, reward, done, False, info

        # Get the original pattern this polygon belongs to
        original_pattern_idx = self.pattern_idx_map[pattern_idx] if pattern_idx < len(self.pattern_idx_map) else pattern_idx
        pattern = self.patterns[original_pattern_idx]
        
        # Try to place pattern using Shapely for precise collision detection
        success = self._place_pattern(pattern, position, rotation, pattern_idx)

        # Track statistics about this placement for better learning
        placement_quality = 0.0
        pattern_area = 0.0

        if success:
            # Calculate reward using enhanced reward function
            reward = self._calculate_reward()

            # Calculate additional metrics for pattern placement quality
            # Measure how well this pattern fits with existing patterns
            if "contours" in pattern and pattern["contours"]:
                try:
                    pattern_area = cv2.contourArea(pattern["contours"][0])
                except:
                    # Fallback to dimensions if contour area calculation fails
                    if "dimensions" in pattern:
                        width, height = pattern["dimensions"]
                        pattern_area = float(width) * float(height)

            # Calculate placement quality metrics
            # Higher value indicates better placement
            placement_quality = self._calculate_placement_quality(pattern_idx, position, rotation)

            # Add the placement to tracking
            self.placed_patterns.append(
                {
                    "pattern": pattern,
                    "position": position,
                    "rotation": rotation,
                    "pattern_idx": pattern_idx,
                    "quality": placement_quality,
                    "area": pattern_area,
                }
            )
            self.available_patterns.remove(pattern_idx)
        else:
            # Enhanced penalty for invalid placement that gets worse as we place more patterns
            # This encourages the agent to make valid placements, especially in later stages
            progress = len(self.placed_patterns) / max(1, len(self.patterns))
            reward = -1.0 - progress  # Penalty increases as more patterns are placed

        # Update state
        self.current_state = self._get_state()

        # Check if episode is done
        terminated = len(self.placed_patterns) == len(self.patterns)

        # Add a maximum step limit to prevent very long episodes
        max_steps = len(self.patterns) * 3  # Allow multiple attempts per pattern
        truncated = len(self.placed_patterns) > max_steps  # Truncate if too many steps

        # Enhanced info dictionary with more detailed metrics
        info = {
            "success": success,
            "utilization": self._calculate_utilization(),
            "remaining_patterns": len(self.patterns) - len(self.placed_patterns),
            "placement_quality": placement_quality,
            "compactness": self._calculate_compactness(),
            "edge_distance": self._calculate_edge_distance(),
            "pattern_distribution": self._calculate_pattern_distribution(),
        }

        return self.current_state, reward, terminated, truncated, info

    def _calculate_placement_quality(self, pattern_idx: int, position: Tuple[int, int], rotation: float) -> float:
        """Calculate how good a pattern placement is based on its relationship to other patterns
        
        Evaluates:
        1. Closeness to other patterns (tighter packing)
        2. Alignment with cloth edges and other patterns
        3. Material utilization improvement
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Start with base quality score
        quality = 0.5

        # Nothing to compare against if this is the first pattern
        if not self.placed_patterns:
            return quality

        # Get pattern polygon
        if pattern_idx >= len(self.pattern_polygons) or self.pattern_polygons[pattern_idx] is None:
            return quality

        pattern_polygon = self.pattern_polygons[pattern_idx]

        # Rotate and translate to position
        placed_polygon = sa.rotate(pattern_polygon, rotation, origin=(0, 0))
        placed_polygon = sa.translate(placed_polygon, position[0], position[1])

        # Calculate minimum distance to any other placed pattern
        min_distance = float('inf')
        alignment_score = 0.0

        for placed in self.placed_patterns:
            placed_pattern_idx = placed["pattern_idx"]
            placed_pos = placed["position"]
            placed_rot = placed["rotation"]

            if (placed_pattern_idx < len(self.pattern_polygons) and
                self.pattern_polygons[placed_pattern_idx] is not None):

                # Get the polygon for the placed pattern
                existing_poly = self.pattern_polygons[placed_pattern_idx]
                existing_poly = sa.rotate(existing_poly, placed_rot, origin=(0, 0))
                existing_poly = sa.translate(existing_poly, placed_pos[0], placed_pos[1])

                # Calculate distance between polygons
                try:
                    # Distance between polygons (0 if they touch)
                    distance = placed_polygon.distance(existing_poly)
                    min_distance = min(min_distance, distance)

                    # Check alignment of edges for better packing
                    # Closer to 0/90/180/270 degrees is better alignment
                    angle_diff = abs((rotation - placed_rot) % 90)
                    angle_alignment = 1.0 - (min(angle_diff, 90 - angle_diff) / 45.0)
                    alignment_score += angle_alignment / len(self.placed_patterns)
                except:
                    pass

        # Normalize distance (closer is better)
        max_possible_dist = np.sqrt(self.cloth_width**2 + self.cloth_height**2)
        distance_score = 1.0 - min(min_distance / max_possible_dist, 1.0)

        # Calculate edge alignment with cloth borders
        border_alignment = 0.0

        # Check how close to cloth edges
        left_edge_dist = position[0]
        right_edge_dist = self.cloth_width - position[0]
        top_edge_dist = position[1]
        bottom_edge_dist = self.cloth_height - position[1]

        # Reward being close to edges but not too close
        edge_distances = [left_edge_dist, right_edge_dist, top_edge_dist, bottom_edge_dist]
        for dist in edge_distances:
            if dist < 10:  # Very close to edge
                border_alignment += 0.25

        # Combine scores with different weights
        quality = (
            0.4 * distance_score +   # Closeness to other patterns
            0.3 * alignment_score +  # Alignment with other patterns
            0.3 * border_alignment   # Alignment with cloth edges
        )

        return min(1.0, max(0.0, quality))

    def _place_pattern(
        self, pattern: Dict, position: Tuple[int, int], rotation: float, 
        pattern_polygon_idx: int = None
    ) -> bool:
        """Attempt to place pattern at given position and rotation using Shapely
        
        Args:
            pattern: Dictionary with pattern data
            position: (x, y) position to place the pattern
            rotation: Rotation angle in degrees
            pattern_polygon_idx: Index in pattern_polygons to use (for multi-piece patterns)
        """
        x, y = position
        
        # Use provided polygon index if available
        if pattern_polygon_idx is not None and pattern_polygon_idx < len(self.pattern_polygons):
            polygon_idx = pattern_polygon_idx
        else:
            # Find pattern index by comparing object identities
            polygon_idx = -1
            for i, p in enumerate(self.patterns):
                if p is pattern:
                    # Use the first matching polygon for this pattern
                    for j, mapped_idx in enumerate(self.pattern_idx_map):
                        if mapped_idx == i:
                            polygon_idx = j
                            break
                    if polygon_idx >= 0:
                        break

            # If we couldn't find a polygon, try to use the idx field
            if polygon_idx == -1:
                pattern_idx = pattern.get("idx", -1)
                # Find first polygon that maps to this pattern
                for j, mapped_idx in enumerate(self.pattern_idx_map):
                    if mapped_idx == pattern_idx:
                        polygon_idx = j
                        break

        # If we still couldn't find a valid polygon, return failure
        if polygon_idx == -1 or polygon_idx >= len(self.pattern_polygons):
            return False

        if self.pattern_polygons[polygon_idx] is None:
            return False

        # Get the pattern polygon
        pattern_polygon = self.pattern_polygons[polygon_idx]

        # Rotate the polygon
        rotated_polygon = sa.rotate(pattern_polygon, rotation, origin=(0, 0))

        # Translate to position
        placed_polygon = sa.translate(rotated_polygon, x, y)

        # Check if pattern is within cloth bounds
        if not self.cloth_polygon.contains(placed_polygon):
            return False

        # Check overlap with already placed patterns
        for placed in self.placed_patterns:
            placed_pattern_idx = placed["pattern_idx"]
            placed_pos = placed["position"]
            placed_rot = placed["rotation"]

            if (
                placed_pattern_idx < len(self.pattern_polygons)
                and self.pattern_polygons[placed_pattern_idx] is not None
            ):
                existing_poly = self.pattern_polygons[placed_pattern_idx]
                existing_poly = sa.rotate(existing_poly, placed_rot, origin=(0, 0))
                existing_poly = sa.translate(
                    existing_poly, placed_pos[0], placed_pos[1]
                )

                if placed_polygon.intersects(existing_poly):
                    return False

        # Update the cloth space raster representation for visualization
        pattern_mask = np.zeros_like(self.cloth_space)

        # Convert Shapely polygon to OpenCV contour format
        polygon_points = np.array(placed_polygon.exterior.coords).astype(int)[
            :-1
        ]  # Remove repeated last point
        cv2.fillPoly(pattern_mask, [polygon_points], 1)

        # Place pattern
        self.cloth_space = np.logical_or(self.cloth_space, pattern_mask).astype(
            np.uint8
        )

        return True

    def _calculate_reward(self) -> float:
        """Calculate reward based on multiple factors

        Returns:
            Combined reward value
        """
        utilization = self._calculate_utilization()
        compactness = self._calculate_compactness()
        edge_distance = self._calculate_edge_distance()
        pattern_distribution = self._calculate_pattern_distribution()
        valid_placement = 1.0  # All placements are valid at this point

        # Weighted reward calculation with enhanced pattern distribution
        reward = (
            0.4 * utilization
            + 0.3 * compactness
            + 0.1 * edge_distance
            + 0.1 * pattern_distribution
            + 0.1 * valid_placement
        )

        return reward

    def _calculate_pattern_distribution(self) -> float:
        """Calculate how well patterns are distributed across the cloth
        
        This rewards placements that utilize different areas of the cloth
        rather than clustering all patterns in one area.
        
        Returns:
            Distribution score between 0.0 and 1.0
        """
        if not self.placed_patterns:
            return 0.0

        # Divide cloth into a grid of cells
        grid_size = 8  # 8x8 grid
        cell_width = self.cloth_width / grid_size
        cell_height = self.cloth_height / grid_size

        # Track which cells are occupied
        occupied_cells = set()

        # Count occupied cells
        for placement in self.placed_patterns:
            pos_x, pos_y = placement["position"]

            # Calculate which cell this pattern's position falls into
            cell_x = min(grid_size - 1, max(0, int(pos_x / cell_width)))
            cell_y = min(grid_size - 1, max(0, int(pos_y / cell_height)))

            # Add to occupied cells
            occupied_cells.add((cell_x, cell_y))

        # Calculate distribution score
        # Normalize by total possible cells for better scaling with few patterns
        max_occupied = min(len(self.placed_patterns) * 2, grid_size * grid_size)
        distribution_score = len(occupied_cells) / max_occupied

        return distribution_score

    def _calculate_utilization(self) -> float:
        """Calculate material utilization percentage

        Returns:
            Ratio of used area to total area
        """
        used_area = np.sum(self.cloth_space)
        total_area = self.cloth_width * self.cloth_height
        return used_area / total_area

    def _calculate_compactness(self) -> float:
        """Calculate how compactly patterns are placed

        Returns:
            Ratio of used area to bounding box area
        """
        if np.sum(self.cloth_space) == 0:
            return 0.0

        rows = np.any(self.cloth_space, axis=1)
        cols = np.any(self.cloth_space, axis=0)

        height = np.sum(rows)
        width = np.sum(cols)

        used_area = np.sum(self.cloth_space)
        bounding_area = height * width

        return used_area / bounding_area if bounding_area > 0 else 0.0

    def _calculate_edge_distance(self) -> float:
        """Calculate average distance to cloth edges

        Returns:
            Normalized average distance to edges
        """
        if not self.placed_patterns:
            return 0.0

        total_distance = 0
        for placement in self.placed_patterns:
            pos_x, pos_y = placement["position"]
            distance_to_edge = min(
                pos_x, pos_y, self.cloth_width - pos_x, self.cloth_height - pos_y
            )
            total_distance += distance_to_edge

        return 1.0 - (
            total_distance
            / (len(self.placed_patterns) * max(self.cloth_width, self.cloth_height))
        )

    def _get_state(self) -> np.ndarray:
        """Get current state representation with enhanced pattern information
        
        Enhanced state representation with:
        1. Current cloth space (occupied areas)
        2. Next pattern to place
        3. Distance transform for spatial awareness
        4. Individual channels for all available patterns to enable planning
        
        Returns:
            State as stacked channels
        """
        try:
            # Get core channels ensuring consistent types
            cloth_space = self.cloth_space.astype(np.float32)
            pattern_channel = self._get_pattern_channel().astype(np.float32)
            distance_channel = self._get_distance_channel().astype(np.float32)

            # Make sure all channels have the same shape
            height, width = cloth_space.shape

            # Ensure pattern channel has the same shape
            if pattern_channel.shape != cloth_space.shape:
                logger.warning(
                    f"Pattern channel shape mismatch: {pattern_channel.shape} vs {cloth_space.shape}"
                )
                pattern_channel = np.zeros_like(cloth_space, dtype=np.float32)

            # Ensure distance channel has the same shape
            if distance_channel.shape != cloth_space.shape:
                logger.warning(
                    f"Distance channel shape mismatch: {distance_channel.shape} vs {cloth_space.shape}"
                )
                distance_channel = np.zeros_like(cloth_space, dtype=np.float32)

            # Create channels for all patterns to help with planning
            pattern_channels = []

            # First add the core channels
            pattern_channels = [cloth_space, pattern_channel, distance_channel]

            # Then add a channel for each available pattern
            for pattern_idx in self.available_patterns:
                if pattern_idx < len(self.patterns):
                    pattern = self.patterns[pattern_idx]
                    pattern_mask = np.zeros_like(cloth_space)

                    # Draw the pattern using its contours or dimensions
                    if "contours" in pattern and pattern["contours"]:
                        try:
                            # Make sure contours are properly formatted for fillPoly
                            contours_copy = []
                            for contour in pattern["contours"]:
                                if contour is not None and len(contour) > 0:
                                    # Ensure contour is int32 and has correct shape
                                    contour_copy = np.array(contour, dtype=np.int32)
                                    if len(contour_copy.shape) == 3 and contour_copy.shape[1] == 1:
                                        # Reshape from (n, 1, 2) to (n, 2) if needed
                                        contour_copy = contour_copy.reshape((contour_copy.shape[0], 2))
                                    contours_copy.append(contour_copy)
                            
                            if contours_copy:
                                cv2.fillPoly(pattern_mask, contours_copy, 1)
                            else:
                                # If no valid contours, use fallback
                                self._fallback_rectangle(pattern_mask, pattern)
                        except Exception as e:
                            logger.warning(f"Error drawing pattern mask: {e}")
                            self._fallback_rectangle(pattern_mask, pattern)
                    else:
                        self._fallback_rectangle(pattern_mask, pattern)

                    pattern_channels.append(pattern_mask)

            # Pad with empty channels if we have fewer patterns than expected
            expected_channels = 3 + len(self.patterns)
            while len(pattern_channels) < expected_channels:
                pattern_channels.append(np.zeros_like(cloth_space))

            # Stack all channels
            state = np.stack(pattern_channels).astype(np.float32)

            return state
        except Exception as e:
            logger.warning(f"Error creating state: {e}")
            # Return a default state of zeros (3 + number of patterns channels)
            shape = self.cloth_space.shape
            return np.zeros((3 + len(self.patterns), shape[0], shape[1]), dtype=np.float32)

    def _get_pattern_channel(self) -> np.ndarray:
        """Create channel showing next pattern to place"""

        if self.available_patterns:
            # Get the first available pattern index
            next_pattern_idx = self.available_patterns[0]
            next_pattern = self.patterns[next_pattern_idx]
            pattern_channel = np.zeros_like(self.cloth_space)

            if "contours" in next_pattern and next_pattern["contours"]:
                # Try to use contours
                try:
                    # Make sure contours are properly formatted for fillPoly
                    contours_copy = []
                    for contour in next_pattern["contours"]:
                        if contour is not None and len(contour) > 0:
                            # Ensure contour is int32 and has correct shape
                            contour_copy = np.array(contour, dtype=np.int32)
                            if len(contour_copy.shape) == 3 and contour_copy.shape[1] == 1:
                                # Reshape from (n, 1, 2) to (n, 2) if needed
                                contour_copy = contour_copy.reshape((contour_copy.shape[0], 2))
                            contours_copy.append(contour_copy)
                    
                    if contours_copy:
                        cv2.fillPoly(pattern_channel, contours_copy, 1)
                    else:
                        # If no valid contours, use fallback
                        self._fallback_rectangle(pattern_channel, next_pattern)
                except Exception as e:
                    logger.warning(
                        f"Error filling pattern polygon: {e}, using shapely fallback"
                    )
                    # Use shapely for fallback if available
                    if (
                        next_pattern_idx < len(self.pattern_polygons)
                        and self.pattern_polygons[next_pattern_idx] is not None
                    ):
                        polygon = self.pattern_polygons[next_pattern_idx]
                        # Center in the middle of the cloth
                        center_x, center_y = (
                            self.cloth_width // 2,
                            self.cloth_height // 2,
                        )
                        # Get pattern centroid
                        centroid = polygon.centroid
                        # Translate to center
                        centered_polygon = sa.translate(
                            polygon, center_x - centroid.x, center_y - centroid.y
                        )
                        # Convert to numpy array for drawing
                        polygon_points = np.array(
                            centered_polygon.exterior.coords
                        ).astype(int)[:-1]
                        cv2.fillPoly(pattern_channel, [polygon_points], 1)
                    else:
                        self._fallback_rectangle(pattern_channel, next_pattern)
            else:
                # Fallback to rectangle if no contours
                self._fallback_rectangle(pattern_channel, next_pattern)

            return pattern_channel

        return np.zeros_like(self.cloth_space)

    def _fallback_rectangle(self, pattern_channel: np.ndarray, pattern: Dict) -> None:
        """Draw a fallback rectangle on the pattern channel"""
        try:
            if "dimensions" in pattern:
                # Get dimensions (handle both tensor and numpy types)
                dimensions = pattern["dimensions"]
                if hasattr(dimensions, "tolist"):  # Handle torch tensors
                    dimensions = dimensions.tolist()

                width, height = map(int, dimensions)
                # Ensure positive dimensions
                width, height = max(5, width), max(5, height)
                # Place at center
                center_x = self.cloth_width // 2
                center_y = self.cloth_height // 2
                x1, y1 = center_x - width // 2, center_y - height // 2
                x2, y2 = x1 + width, y1 + height
                # Ensure coordinates are within cloth boundaries
                x1 = max(0, min(x1, self.cloth_width - 10))
                y1 = max(0, min(y1, self.cloth_height - 10))
                x2 = max(x1 + 5, min(x2, self.cloth_width))
                y2 = max(y1 + 5, min(y2, self.cloth_height))
                cv2.rectangle(pattern_channel, (x1, y1), (x2, y2), 1, -1)
            else:
                # If no dimensions available, draw a small rectangle at center
                center_x, center_y = self.cloth_width // 2, self.cloth_height // 2
                size = min(self.cloth_width, self.cloth_height) // 10
                cv2.rectangle(
                    pattern_channel,
                    (center_x - size, center_y - size),
                    (center_x + size, center_y + size),
                    1,
                    -1,
                )
        except Exception as e:
            logger.warning(f"Error creating fallback rectangle: {e}")
            # Last resort: place a small square in the center
            center_x, center_y = self.cloth_width // 2, self.cloth_height // 2
            cv2.rectangle(
                pattern_channel,
                (center_x - 10, center_y - 10),
                (center_x + 10, center_y + 10),
                1,
                -1,
            )

    def _get_distance_channel(self) -> np.ndarray:
        """Create distance transform channel"""

        try:
            # Distance transform helps the model understand spatial relationships
            distance_map = cv2.distanceTransform(
                (1 - self.cloth_space).astype(np.uint8), cv2.DIST_L2, 5
            )

            # Normalize the distance values to 0-1 range
            if np.max(distance_map) > 0:
                distance_map = distance_map / np.max(distance_map)

            return distance_map
        except Exception as e:
            logger.warning(f"Error creating distance transform: {e}")
            # Return zeros if distance transform fails
            return np.zeros_like(self.cloth_space, dtype=np.float32)

    def render(self):
        """Render the current state of the environment"""

        if self.render_mode is None:
            return None

        import matplotlib.pyplot as plt

        # Create a figure with the cloth space
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(self.cloth_space, cmap="gray")
        plt.title(f"Cloth Space (Utilization: {self._calculate_utilization():.4f})")

        if self.render_mode == "rgb_array":
            # Return an RGB array
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            return img
        elif self.render_mode == "human":
            # Display the figure
            plt.show()
            plt.close(fig)
            return None


class ManagerNetwork(nn.Module):
    """High-level network for pattern selection

    This network determines which pattern to place next based on the
    current state of the cloth space and available patterns.
    
    Enhanced to better understand pattern relationships and optimize placement sequence.
    """

    def __init__(self, input_channels: int, hidden_dim: int, num_patterns: int):
        """Initialize the manager network

        Args:
            input_channels: Number of input channels in the state
            hidden_dim: Size of hidden layer
            num_patterns: Number of patterns to choose from
        """
        super().__init__()

        # Convolutional layers for processing spatial information with more filters
        # and better feature extraction by using deeper network
        self.conv = nn.Sequential(
            # First conv block with more filters
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second conv block with deeper features
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Adaptive pooling to handle variable sized inputs
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # Improved fully connected layers for pattern selection
        # with attention to pattern relationships
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_patterns),
        )

        # Global attention mechanism to focus on important cloth areas
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, 1),  # 1x1 conv to produce attention map
            nn.Sigmoid()  # Normalize attention weights
        )

    def forward(self, x):
        """Forward pass through the network with attention mechanism

        Args:
            x: Input state tensor

        Returns:
            Pattern selection probabilities
        """
        # Extract features through convolutional layers
        features = self.conv(x)

        # Apply attention mechanism
        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        # Flatten and pass through fully connected layers
        x = attended_features.view(attended_features.size(0), -1)
        sequence_probs = F.softmax(self.fc(x), dim=1)
        return sequence_probs


class WorkerNetwork(nn.Module):
    """Low-level network for pattern placement

    This network determines where and at what rotation to place a selected pattern.
    
    Enhanced with deeper feature extraction and advanced placement heatmap generation
    to better optimize pattern placement for material utilization.
    """

    def __init__(
        self, input_channels: int, num_rotations: int, cloth_dims: Tuple[int, int]
    ):
        """Initialize the worker network

        Args:
            input_channels: Number of input channels in the state
            num_rotations: Number of possible rotation angles
            cloth_dims: Dimensions of the cloth (height, width)
        """
        super().__init__()

        self.cloth_height, self.cloth_width = cloth_dims

        # Enhanced convolutional backbone for better spatial understanding
        # Using ResNet-like skip connections for better gradient flow
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Global pooling to get fixed-size feature regardless of input dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Enhanced placement head: U-Net style decoder with skip connections
        # for more precise placement predictions with spatial context awareness
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2),  # 256 = 128 + 128 (skip connection)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Final upsampling to cloth dimensions with refinement
        self.placement_head = nn.Sequential(
            nn.Upsample(
                size=(self.cloth_height, self.cloth_width),
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(128, 64, 3, padding=1),  # 128 = 64 + 64 (skip connection)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),  # Final 1x1 conv for heatmap
        )

        # Enhanced rotation head with deeper MLP
        self.rotation_head = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_rotations),
        )

        # Attention mechanism for emphasizing important regions
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass through the network with skip connections

        Args:
            x: Input state tensor with pattern

        Returns:
            Tuple of (placement_map, rotation_probabilities)
        """
        # Extract features with skip connections for U-Net style architecture
        conv1_features = self.conv1(x)
        conv2_features = self.conv2(conv1_features)
        conv3_features = self.conv3(conv2_features)

        # Apply attention to highest level features
        attention_weights = self.attention(conv3_features)
        attended_features = conv3_features * attention_weights

        # Feature extraction for rotation prediction
        pooled_features = self.global_pool(attended_features)
        flat_features = pooled_features.view(pooled_features.size(0), -1)

        # Generate rotation probabilities
        rotation_logits = self.rotation_head(flat_features)
        rotation_probs = F.softmax(rotation_logits, dim=1)

        # Generate placement heatmap using U-Net style decoder with skip connections
        up1 = self.upsample1(attended_features)
        
        # Check and adjust feature dimensions if necessary
        # Fix for tensor size mismatch: Expected size 250 but got size 251
        if up1.size(2) != conv2_features.size(2) or up1.size(3) != conv2_features.size(3):
            up1 = F.interpolate(up1, size=(conv2_features.size(2), conv2_features.size(3)), 
                                mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection from encoder
        cat1 = torch.cat([up1, conv2_features], dim=1)

        up2 = self.upsample2(cat1)
        
        # Also ensure the second upsampling matches dimensions
        if up2.size(2) != conv1_features.size(2) or up2.size(3) != conv1_features.size(3):
            up2 = F.interpolate(up2, size=(conv1_features.size(2), conv1_features.size(3)), 
                               mode='bilinear', align_corners=False)
            
        # Concatenate with skip connection from encoder
        cat2 = torch.cat([up2, conv1_features], dim=1)

        # Final placement map
        placement_map = self.placement_head(cat2)

        return placement_map, rotation_probs


class HierarchicalRL:
    """Hierarchical Reinforcement Learning for pattern packing

    This class implements a two-level hierarchical reinforcement learning approach
    with a manager network for pattern selection and a worker network for placement.

    References:
    - "Planning Irregular Object Packing via Hierarchical Reinforcement Learning" (Wang et al., 2022)
    - "Tree Search + Reinforcement Learning for Two-Dimensional Cutting Stock Problem
       With Complex Constraints" (Zhang et al., 2023)
    """

    def __init__(self, env: PackingEnvironment, device: torch.device = None):
        """Initialize the hierarchical RL agent

        Args:
            env: Packing environment
            device: Computation device (CPU/GPU)
        """
        self.env = env
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Get cloth dimensions
        cloth_dims = (self.env.cloth_height, self.env.cloth_width)

        # Initialize networks
        # Account for all channels in observation space (3 + number of patterns)
        self.manager = ManagerNetwork(
            input_channels=env.observation_space.shape[0],  # All channels from observation space
            hidden_dim=256,
            num_patterns=len(env.patterns),
        ).to(self.device)

        # The worker network needs to take the state dimensions plus one additional channel
        # for the selected pattern (state + 1)
        self.worker = WorkerNetwork(
            input_channels=env.observation_space.shape[0] + 1,  # State channels + selected pattern
            num_rotations=len(env.rotation_angles),
            cloth_dims=cloth_dims,
        ).to(self.device)

        # Initialize optimizers
        self.manager_optimizer = optim.Adam(self.manager.parameters(), lr=0.001)
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=0.001)

        # Experience buffer for batch learning
        self.buffer = []
        self.gamma = 0.99  # Discount factor

    def train(self, num_episodes: int):
        """Train both networks through reinforcement learning

        Args:
            num_episodes: Number of training episodes

        Returns:
            Tuple of (best_state, best_utilization)
        """
        best_utilization = 0.0
        best_state = None

        for episode in range(num_episodes):
            state, _ = self.env.reset()  # Gymnasium API returns (state, info)
            episode_reward = 0
            done = False

            # Move state to device and correct format
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            while not done:
                # 1. Manager selects which pattern to place next
                sequence_probs = self.manager(state_tensor)
                pattern_idx = self.select_pattern(
                    sequence_probs.squeeze().cpu().detach().numpy()
                )

                if pattern_idx is None:
                    # No more patterns to place
                    logger.info("All patterns placed, ending episode.")
                    done = True
                    break

                # 2. Worker selects where and at what rotation to place the pattern
                # Create worker state by adding pattern channel
                pattern_channel = np.zeros_like(self.env.cloth_space)
                pattern = self.env.patterns[pattern_idx]

                if "contours" in pattern and pattern["contours"]:
                    try:
                        # Make sure contours are properly formatted for fillPoly
                        contours_copy = []
                        for contour in pattern["contours"]:
                            if contour is not None and len(contour) > 0:
                                # Ensure contour is int32 and has correct shape
                                contour_copy = np.array(contour, dtype=np.int32)
                                if len(contour_copy.shape) == 3 and contour_copy.shape[1] == 1:
                                    # Reshape from (n, 1, 2) to (n, 2) if needed
                                    contour_copy = contour_copy.reshape((contour_copy.shape[0], 2))
                                contours_copy.append(contour_copy)
                        
                        if contours_copy:
                            cv2.fillPoly(pattern_channel, contours_copy, 1)
                        else:
                            # Fallback to a simple rectangle if contours are invalid
                            self._fallback_rectangle(pattern_channel, pattern)
                    except Exception as e:
                        logger.warning(f"Error filling pattern: {e}, using fallback")
                        self._fallback_rectangle(pattern_channel, pattern)

                worker_state = np.concatenate(
                    [state, pattern_channel[np.newaxis, :, :]]
                )
                worker_state_tensor = (
                    torch.FloatTensor(worker_state).unsqueeze(0).to(self.device)
                )

                placement_map, rotation_probs = self.worker(worker_state_tensor)

                # 3. Select position and rotation
                # For position, find the pixel with highest probability
                pos_idx = torch.argmax(placement_map.view(-1)).item()
                pos_y, pos_x = np.unravel_index(
                    pos_idx, (self.env.cloth_height, self.env.cloth_width)
                )

                # For rotation, select the rotation angle with highest probability
                rotation_idx = torch.argmax(rotation_probs).item()
                rotation = self.env.rotation_angles[rotation_idx]

                # 4. Take action in environment
                action = {
                    "pattern_idx": pattern_idx,
                    "position": (pos_x, pos_y),
                    "rotation": rotation,
                }

                # Gymnasium API returns 5 values
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 5. Store experience for learning
                self.buffer.append(
                    {
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "done": done,
                        "sequence_probs": sequence_probs.squeeze()
                        .cpu()
                        .detach()
                        .numpy(),
                        "placement_map": placement_map.squeeze().cpu().detach().numpy(),
                        "rotation_probs": rotation_probs.squeeze()
                        .cpu()
                        .detach()
                        .numpy(),
                    }
                )

                # 6. Update networks if buffer has enough samples
                if len(self.buffer) >= 16:  # Batch size
                    self._update_networks()
                    self.buffer = []

                # 7. Update state and running reward
                state = next_state
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                episode_reward += reward

                # Track best state (for visualization and final result)
                current_utilization = info["utilization"]
                if current_utilization > best_utilization:
                    best_utilization = current_utilization
                    best_state = self.env.cloth_space.copy()
                    logger.info(f"New best utilization: {best_utilization:.4f}")

            # Log episode results
            if episode % 10 == 0 or episode == num_episodes - 1:
                logger.info(
                    f"Episode {episode}/{num_episodes}, "
                    f"Reward: {episode_reward:.2f}, "
                    f"Utilization: {info['utilization']:.4f}"
                )

        # Return best placement state achieved during training
        return best_state, best_utilization

    def select_pattern(self, sequence_probs, epsilon=0.1):
        """Select a pattern using advanced selection strategy
        
        Enhanced with pattern size awareness and prioritization to improve
        multi-pattern packing efficiency. Uses an optimized selection strategy
        that considers:
        
        1. Largest patterns first (greedy placement of large patterns)
        2. Probability scores from manager network
        3. Exploration vs. exploitation balance
        
        This approach follows research showing that placing larger patterns first
        often leads to better overall utilization.

        Args:
            sequence_probs: Probability distribution over patterns
            epsilon: Exploration rate

        Returns:
            Selected pattern index or None if no patterns available
        """
        if not self.env.available_patterns:
            return None  # No patterns available

        # Ensure sequence_probs is a numpy array
        if isinstance(sequence_probs, (int, float)):
            # If it's a scalar, convert to array
            sequence_probs = np.array([sequence_probs])

        # Pure exploration path (random selection)
        if np.random.random() < epsilon:
            # Explore: choose a random available pattern
            return np.random.choice(self.env.available_patterns)
        else:
            # Exploitation path with size-aware prioritization
            valid_actions = self.env.available_patterns
            if not valid_actions:
                return None  # No valid actions

            # Special case: only one available pattern
            if sequence_probs.ndim == 0 or (len(sequence_probs) == 1 and len(valid_actions) == 1):
                return valid_actions[0]

            # Get pattern sizes
            pattern_sizes = []
            for pattern_idx in valid_actions:
                pattern = self.env.patterns[pattern_idx]

                # Get contour area if available
                if "contours" in pattern and pattern["contours"]:
                    area = cv2.contourArea(pattern["contours"][0])
                # Otherwise use dimensions to calculate area
                elif "dimensions" in pattern:
                    width, height = pattern["dimensions"]
                    # Handle potentially different types (tensor, numpy, etc.)
                    if hasattr(width, "item"):
                        width, height = width.item(), height.item()
                    area = float(width) * float(height)
                else:
                    # Default area if no size information available
                    area = 0

                pattern_sizes.append((pattern_idx, area))

            # Sort patterns by area (largest first)
            pattern_sizes.sort(key=lambda x: x[1], reverse=True)

            # Filter probabilities for valid actions
            valid_probs = sequence_probs[valid_actions]

            # Check if we have valid probabilities
            if len(valid_probs) > 0:
                # Blend size priority with network prediction
                # Combine size ranking with network probabilities (70% size, 30% network)
                combined_scores = {}

                # Normalize size ranking to 0-1 range
                size_ranks = {}
                for i, (pattern_idx, _) in enumerate(pattern_sizes):
                    size_ranks[pattern_idx] = 1.0 - (i / max(1, len(pattern_sizes) - 1))

                # Combine scores
                for pattern_idx in valid_actions:
                    prob_idx = valid_actions.index(pattern_idx)
                    network_score = valid_probs[prob_idx]
                    size_score = size_ranks.get(pattern_idx, 0.0)

                    # Combined score (weighted blend)
                    combined_scores[pattern_idx] = 0.7 * size_score + 0.3 * network_score

                # Select pattern with highest combined score
                best_action = max(combined_scores.items(), key=lambda x: x[1])[0]
                return best_action
            else:
                # Fallback to largest pattern if probabilities are invalid
                return pattern_sizes[0][0] if pattern_sizes else np.random.choice(valid_actions)

    def _update_networks(self):
        """Update both networks using stored experiences"""
        # Get batch of experiences
        batch = self.buffer  # Use all experiences in buffer

        # Prepare batch data
        states = torch.FloatTensor([x["state"] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x["reward"] for x in batch]).to(self.device)
        actions = [x["action"] for x in batch]

        # Update manager network
        sequence_probs = self.manager(states)

        # Calculate manager loss (policy gradient)
        manager_loss = 0
        for i, action in enumerate(actions):
            # Use log probability of the chosen action multiplied by the reward
            if action["pattern_idx"] < sequence_probs.shape[1]:
                log_prob = torch.log(sequence_probs[i, action["pattern_idx"]] + 1e-10)
                manager_loss -= log_prob * rewards[i]  # Gradient ascent

        manager_loss /= len(batch)  # Normalize by batch size

        self.manager_optimizer.zero_grad()
        manager_loss.backward()
        self.manager_optimizer.step()

        # Update worker network
        # Create worker states by adding pattern channel
        worker_states = []
        for x in batch:
            state = x["state"]
            pattern_idx = x["action"]["pattern_idx"]
            pattern_mask = np.zeros_like(self.env.cloth_space)

            if pattern_idx < len(self.env.patterns):
                pattern = self.env.patterns[pattern_idx]
                if "contours" in pattern and pattern["contours"]:
                    try:
                        # Make sure contours are properly formatted for fillPoly
                        contours_copy = []
                        for contour in pattern["contours"]:
                            if contour is not None and len(contour) > 0:
                                # Ensure contour is int32 and has correct shape
                                contour_copy = np.array(contour, dtype=np.int32)
                                if len(contour_copy.shape) == 3 and contour_copy.shape[1] == 1:
                                    # Reshape from (n, 1, 2) to (n, 2) if needed
                                    contour_copy = contour_copy.reshape((contour_copy.shape[0], 2))
                                contours_copy.append(contour_copy)
                        
                        if contours_copy:
                            cv2.fillPoly(pattern_mask, contours_copy, 1)
                    except Exception as e:
                        logger.warning(f"Error filling pattern in _update_networks: {e}")

            worker_state = np.concatenate([state, pattern_mask[np.newaxis, :, :]])
            worker_states.append(worker_state)

        worker_states_tensor = torch.FloatTensor(worker_states).to(self.device)
        placement_maps, rotation_probs = self.worker(worker_states_tensor)

        # Calculate worker losses
        # 1. Position loss
        placement_targets = []
        for action in actions:
            pos_x, pos_y = action["position"]
            target = pos_y * self.env.cloth_width + pos_x
            placement_targets.append(target)

        placement_targets_tensor = torch.LongTensor(placement_targets).to(self.device)

        placement_loss = F.cross_entropy(
            placement_maps.view(len(batch), -1), placement_targets_tensor
        )

        # 2. Rotation loss
        rotation_targets = [
            self.env.rotation_angles.index(x["action"]["rotation"]) for x in batch
        ]
        rotation_targets_tensor = torch.LongTensor(rotation_targets).to(self.device)

        rotation_loss = F.cross_entropy(rotation_probs, rotation_targets_tensor)

        # Combine worker losses
        worker_loss = placement_loss + rotation_loss

        self.worker_optimizer.zero_grad()
        worker_loss.backward()
        self.worker_optimizer.step()

    def save_model(self, model_path: str):
        """Save trained models

        Args:
            model_path: Path to save the model
        """
        torch.save(
            {"manager": self.manager.state_dict(), "worker": self.worker.state_dict()},
            model_path,
        )
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load trained models

        Args:
            model_path: Path to load the model from
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.manager.load_state_dict(checkpoint["manager"])
        self.worker.load_state_dict(checkpoint["worker"])
        logger.info(f"Model loaded from {model_path}")

    def infer(self, visualize=False):
        """Run inference to find optimal pattern placement

        Args:
            visualize: Whether to generate visualizations

        Returns:
            Tuple of (final_state, utilization, placement_data)
        """
        state, _ = self.env.reset()  # Gymnasium API returns (state, info)
        done = False
        total_reward = 0

        # Prepare for visualization if requested
        if visualize:
            import matplotlib.pyplot as plt

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(self.env.cloth_space, cmap="gray")
            plt.title("Initial cloth")
            plt.savefig(os.path.join(output_dir, "initial_cloth.png"))
            plt.close()

        step = 0
        while not done:
            # Move state to device
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get pattern selection from manager (no exploration during inference)
            with torch.no_grad():
                sequence_probs = self.manager(state_tensor)

            # Select best pattern without exploration
            pattern_idx = self.select_pattern(
                sequence_probs.squeeze().cpu().numpy(), epsilon=0
            )

            if pattern_idx is None:
                # No more patterns to place
                break

            # Get placement from worker
            pattern_channel = np.zeros_like(self.env.cloth_space)
            pattern = self.env.patterns[pattern_idx]

            if "contours" in pattern and pattern["contours"]:
                try:
                    # Make sure contours are properly formatted for fillPoly
                    contours_copy = []
                    for contour in pattern["contours"]:
                        if contour is not None and len(contour) > 0:
                            # Ensure contour is int32 and has correct shape
                            contour_copy = np.array(contour, dtype=np.int32)
                            if len(contour_copy.shape) == 3 and contour_copy.shape[1] == 1:
                                # Reshape from (n, 1, 2) to (n, 2) if needed
                                contour_copy = contour_copy.reshape((contour_copy.shape[0], 2))
                            contours_copy.append(contour_copy)
                    
                    if contours_copy:
                        cv2.fillPoly(pattern_channel, contours_copy, 1)
                    else:
                        # If no valid contours, use fallback
                        self._fallback_rectangle(pattern_channel, pattern)
                except Exception as e:
                    logger.warning(f"Error filling pattern: {e}, using fallback")
                    self._fallback_rectangle(pattern_channel, pattern)

            worker_state = np.concatenate([state, pattern_channel[np.newaxis, :, :]])
            worker_state_tensor = (
                torch.FloatTensor(worker_state).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                placement_map, rotation_probs = self.worker(worker_state_tensor)

            # Select position and rotation
            pos_idx = torch.argmax(placement_map.view(-1)).item()
            pos_y, pos_x = np.unravel_index(
                pos_idx, (self.env.cloth_height, self.env.cloth_width)
            )

            rotation_idx = torch.argmax(rotation_probs).item()
            rotation = self.env.rotation_angles[rotation_idx]

            # Take action
            action = {
                "pattern_idx": pattern_idx,
                "position": (pos_x, pos_y),
                "rotation": rotation,
            }

            # Gymnasium API returns 5 values
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Visualize step if requested
            if visualize:
                plt.figure(figsize=(10, 10))
                plt.imshow(self.env.cloth_space, cmap="gray")
                plt.title(f"Step {step}: Pattern {pattern_idx} placed")
                plt.savefig(os.path.join(output_dir, f"step_{step}_cloth.png"))
                plt.close()

            state = next_state
            step += 1

            # Log progress
            logger.info(
                f"Step {step}: Placed pattern {pattern_idx} at {(pos_x, pos_y)} with rotation {rotation}°"
            )
            logger.info(f"Current utilization: {info['utilization']:.4f}")

        # Final visualization
        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.env.cloth_space, cmap="gray")
            plt.title(f"Final placement: Utilization {info['utilization']:.4f}")
            plt.savefig(os.path.join(output_dir, "final_cloth.png"))
            plt.close()

        logger.info(
            f"Inference complete: Utilization {info['utilization']:.4f}, Reward {total_reward:.2f}"
        )
        return self.env.cloth_space, info["utilization"], self.env.placed_patterns


class PatternFittingModule:
    """Module for fitting patterns onto cloth using Hierarchical RL

    This module provides the main interface for using Hierarchical RL
    to optimize pattern placement on cloth materials.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize pattern fitting module

        Args:
            model_path: Path to load a pretrained model (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.rl_agent = None
        self.rotation_angles = [
            0,
            45,
            90,
            135,
            180,
            225,
            270,
            315,
        ]  # Possible rotation angles

        logger.info(f"Pattern Fitting Module initialized on device: {self.device}")

    def prepare_data(
        self, cloth_data: Dict, patterns_data: List[Dict]
    ) -> Tuple[Dict, List[Dict]]:
        """Prepare cloth and patterns data for packing environment
        
        Enhanced to use cloth and pattern masks for better fitting.

        Args:
            cloth_data: Dictionary containing cloth properties
            patterns_data: List of dictionaries containing pattern properties

        Returns:
            Processed cloth and patterns data
        """
        # Process cloth data - ensure dimensions are reasonable
        processed_cloth = cloth_data.copy()

        # Check if we have a cloth mask to use (from the improved cloth recognition)
        cloth_mask = processed_cloth.get("cloth_mask", None)

        # If we have a cloth mask, use it to get better contours
        if cloth_mask is not None and np.sum(cloth_mask) > 0:
            try:
                # Find contours in the mask
                contours, _ = cv2.findContours(
                    cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # If we found significant contours, use them
                if contours and len(contours) > 0:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 500:  # Reasonable size
                        processed_cloth["contours"] = [largest_contour]

                        # Update dimensions from mask contour
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        dimensions = np.array([w, h], dtype=np.float32)
                        if w > 10 and h > 10:  # Reasonable size check
                            processed_cloth["dimensions"] = dimensions
                            logger.info(f"Using cloth dimensions from mask: {dimensions}")
            except Exception as e:
                logger.warning(f"Error processing cloth mask: {e}")

        # Fix cloth dimensions if they're negative or too small
        dimensions = processed_cloth.get("dimensions", None)
        if dimensions is None or np.any(np.array(dimensions) < 10):
            # Try to get dimensions from contours
            if "contours" in processed_cloth and processed_cloth["contours"]:
                # Use contour bounds
                contour = processed_cloth["contours"][0]
                x, y, w, h = cv2.boundingRect(contour)
                dimensions = np.array([w, h], dtype=np.float32)

            # Use image dimensions if available
            if "image_path" in processed_cloth:
                try:
                    image = cv2.imread(processed_cloth["image_path"])
                    if image is not None:
                        h, w = image.shape[:2]
                        dimensions = np.array([w, h], dtype=np.float32)
                except:
                    pass

            # If all else fails, use default dimensions
            if dimensions is None or np.any(np.array(dimensions) < 10):
                dimensions = np.array([512.0, 512.0], dtype=np.float32)

            processed_cloth["dimensions"] = dimensions
            logger.info(f"Using cloth dimensions: {dimensions}")

        # Make sure dimensions are positive
        cloth_dims = np.abs(np.array(processed_cloth["dimensions"], dtype=np.float32))
        processed_cloth["dimensions"] = tuple(cloth_dims)

        # Scale cloth dimensions if needed (for very large cloth)
        cloth_width, cloth_height = cloth_dims
        max_size = 512  # Maximum size for either dimension

        if cloth_width > max_size or cloth_height > max_size:
            scale = max_size / max(cloth_width, cloth_height)
            processed_cloth["dimensions"] = (cloth_width * scale, cloth_height * scale)
            logger.info(
                f"Scaled cloth from {(cloth_width, cloth_height)} to {processed_cloth['dimensions']}"
            )

            # Also scale the contours if they exist
            if "contours" in processed_cloth and processed_cloth["contours"]:
                try:
                    for i, contour in enumerate(processed_cloth["contours"]):
                        processed_cloth["contours"][i] = contour * scale
                except Exception as e:
                    logger.warning(f"Error scaling contours: {e}")

        # Process patterns data
        processed_patterns = []
        for pattern in patterns_data:
            try:
                # Create a copy of the pattern data
                processed_pattern = pattern.copy()

                # Ensure pattern has contours - use enhanced contours if available
                if (
                    "contours" not in processed_pattern
                    or not processed_pattern["contours"]
                    or len(processed_pattern["contours"]) == 0
                ):
                    # Try to generate contours from dimensions if available
                    if "dimensions" in processed_pattern:
                        # Create a rectangle contour from dimensions
                        dims = processed_pattern["dimensions"]
                        if hasattr(dims, "tolist"):  # Handle torch tensors
                            dims = dims.tolist()
                        width, height = map(float, dims)

                        # Ensure minimum size
                        width, height = max(5.0, width), max(5.0, height)

                        # Create rectangle contour
                        rect = np.array(
                            [[[0, 0]], [[width, 0]], [[width, height]], [[0, height]]],
                            dtype=np.float32,
                        )

                        processed_pattern["contours"] = [rect]
                    else:
                        # Skip pattern with no contours and no dimensions
                        logger.warning(
                            "Pattern missing both contours and dimensions, skipping"
                        )
                        continue
                else:
                    # Filter contours to ensure they're valid
                    valid_contours = []
                    for contour in processed_pattern["contours"]:
                        # Check if contour has enough points and reasonable area
                        if len(contour) >= 3 and cv2.contourArea(contour) > 100:
                            valid_contours.append(contour)

                    # If we have multiple valid contours, use the largest one as the main pattern
                    if valid_contours:
                        if len(valid_contours) > 1:
                            main_contour = max(valid_contours, key=cv2.contourArea)
                            processed_pattern["contours"] = [main_contour]
                        else:
                            processed_pattern["contours"] = valid_contours

                        # Update dimensions from contour if needed
                        contour = processed_pattern["contours"][0]
                        x, y, w, h = cv2.boundingRect(contour)
                        if w > 5 and h > 5:  # Reasonable size check
                            processed_pattern["dimensions"] = np.array([w, h], dtype=np.float32)
                    else:
                        # Handle case where no valid contours were found
                        if "dimensions" in processed_pattern:
                            # Create a rectangle contour from dimensions
                            dims = processed_pattern["dimensions"]
                            if hasattr(dims, "tolist"):  # Handle torch tensors
                                dims = dims.tolist()
                            width, height = map(float, dims)

                            # Ensure minimum size
                            width, height = max(5.0, width), max(5.0, height)

                            # Create rectangle contour
                            rect = np.array(
                                [[[0, 0]], [[width, 0]], [[width, height]], [[0, height]]],
                                dtype=np.float32,
                            )

                            processed_pattern["contours"] = [rect]
                        else:
                            # Skip pattern with no valid geometry data
                            logger.warning("Pattern has no valid contours or dimensions, skipping")
                            continue

                # Apply same scale to patterns if cloth was scaled
                if cloth_width > max_size or cloth_height > max_size:
                    try:
                        for i, contour in enumerate(processed_pattern["contours"]):
                            processed_pattern["contours"][i] = contour * scale
                    except Exception as e:
                        logger.warning(f"Error scaling pattern contours: {e}")

                    # Scale dimensions if they exist
                    if "dimensions" in processed_pattern:
                        try:
                            dims = processed_pattern["dimensions"]
                            if hasattr(dims, "tolist"):  # Handle torch tensors
                                dims = dims.tolist()
                            processed_pattern["dimensions"] = tuple(
                                d * scale for d in dims
                            )
                        except Exception as e:
                            logger.warning(f"Error scaling pattern dimensions: {e}")

                processed_patterns.append(processed_pattern)
            except Exception as e:
                logger.warning(f"Error processing pattern: {e}")

        logger.info(f"Prepared {len(processed_patterns)} patterns for fitting")
        return processed_cloth, processed_patterns

    def train(
        self, cloth_data: Dict, patterns_data: List[Dict], num_episodes: int = 100
    ) -> Dict:
        """Train the pattern fitting model

        Args:
            cloth_data: Dictionary containing cloth properties
            patterns_data: List of dictionaries containing pattern properties
            num_episodes: Number of training episodes

        Returns:
            Dictionary with training results
        """
        # Prepare data for environment
        processed_cloth, processed_patterns = self.prepare_data(
            cloth_data, patterns_data
        )

        # Create packing environment
        env = PackingEnvironment(
            cloth_data=processed_cloth,
            patterns=processed_patterns,
            rotation_angles=self.rotation_angles,
        )

        # Create RL agent
        self.rl_agent = HierarchicalRL(env, self.device)

        # Train the agent
        logger.info(f"Starting training for {num_episodes} episodes")
        best_state, best_utilization = self.rl_agent.train(num_episodes)

        # Save the trained model if path is specified
        if self.model_path:
            self.rl_agent.save_model(self.model_path)

        return {
            "best_state": best_state,
            "best_utilization": best_utilization,
            "training_episodes": num_episodes,
        }

    def fit_patterns(
        self, cloth_data: Dict, patterns_data: List[Dict], visualize: bool = False
    ) -> Dict:
        """Fit patterns onto cloth using trained model

        Args:
            cloth_data: Dictionary containing cloth properties
            patterns_data: List of dictionaries containing pattern properties
            visualize: Whether to generate visualization images

        Returns:
            Dictionary with fitting results
        """
        # Prepare data for environment
        processed_cloth, processed_patterns = self.prepare_data(
            cloth_data, patterns_data
        )

        # Create packing environment
        env = PackingEnvironment(
            cloth_data=processed_cloth,
            patterns=processed_patterns,
            rotation_angles=self.rotation_angles,
        )

        # Create or load RL agent
        if self.rl_agent is None:
            self.rl_agent = HierarchicalRL(env, self.device)

            # Load pretrained model if available
            if self.model_path and os.path.exists(self.model_path):
                self.rl_agent.load_model(self.model_path)
                logger.info(f"Loaded trained model from {self.model_path}")
            else:
                # Train briefly if no model is available
                logger.warning(
                    "No pretrained model available, training for 20 episodes"
                )
                self.rl_agent.train(20)

        # Run inference to get optimal pattern placement
        final_state, utilization, placement_data = self.rl_agent.infer(visualize)

        # Convert placement data to more user-friendly format
        placements = []
        for placement in placement_data:
            placements.append(
                {
                    "pattern_id": placement["pattern_idx"],
                    "position": placement["position"],
                    "rotation": placement["rotation"],
                }
            )

        return {
            "final_state": final_state,  # The cloth space with patterns placed
            "utilization": utilization,  # Material utilization percentage
            "placements": placements,  # Details of each pattern placement
            "cloth_dims": (env.cloth_width, env.cloth_height),
            "patterns": processed_patterns,
            "method": "hierarchical_rl",
        }

    def visualize_result(self, result: Dict, cloth_image=None, save_path: str = None):
        """Visualize pattern fitting result

        Args:
            result: Result dictionary from fit_patterns
            cloth_image: Original cloth image (optional)
            save_path: Path to save visualization (optional)
        """
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create colormap for different patterns - distinct colors for each pattern
        colors = [
            "gray",
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "pink",
            "cyan",
            "magenta",
            "yellow",
        ]
        cmap = ListedColormap(["white"] + colors[: len(result["placements"])])

        # Create a visualization canvas
        canvas = result.get("final_state", np.zeros((400, 400), dtype=int))
        cloth_width, cloth_height = result.get(
            "cloth_dims", (canvas.shape[1], canvas.shape[0])
        )

        # Plot the original cloth if provided
        if cloth_image is not None:
            plt.subplot(1, 2, 1)
            plt.imshow(cloth_image)
            plt.title("Original Cloth")
            plt.axis("off")

            plt.subplot(1, 2, 2)

        # Display the final result
        plt.imshow(canvas, cmap=cmap, interpolation="nearest")
        plt.title(f"Pattern Placement (Utilization: {result['utilization']:.1%})")
        plt.axis("off")

        # Add a legend for each pattern
        handles = []
        for i, placement in enumerate(result["placements"]):
            pattern_idx = placement["pattern_id"]
            handles.append(
                patches.Patch(
                    color=colors[i % len(colors)], label=f"Pattern {pattern_idx}"
                )
            )

        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(
                os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True,
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization saved to {save_path}")

        # Show the plot
        plt.show()
        plt.close()
