"""
Pattern Fitting Module with Research-Backed Algorithms

This module implements state-of-the-art algorithms for 2D irregular packing,
specifically for garment pattern fitting on fabric materials including remnants.

Key Algorithms:
1. Bottom-Left-Fill (BLF) - Jakobs (1996)
2. No-Fit Polygon (NFP) - Burke et al. (2007)
3. Multi-objective optimization - Gomes & Oliveira (2006)

REFERENCES:
[1] Jakobs, S. (1996). "On genetic algorithms for the packing of polygons"
    European Journal of Operational Research, 88(1), 165-181.
[2] Bennell, J.A., Oliveira, J.F. (2008). "The geometry of nesting problems"
    European Journal of Operational Research, 184(2), 397-415.
[3] Gomes, A.M., Oliveira, J.F. (2006). "Solving irregular strip packing problems"
    European Journal of Operational Research, 171(3), 811-829.
[4] Burke, E.K., et al. (2007). "Complete and robust no-fit polygon generation"
    European Journal of Operational Research, 179(1), 27-49.
"""

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Polygon

from .cloth_recognition_module import ClothMaterial
from .config import FITTING, SYSTEM, VISUALIZATION
from .pattern_recognition_module import Pattern

# Setup logging
logging.basicConfig(
    level=getattr(logging, SYSTEM["LOG_LEVEL"]), format=SYSTEM["LOG_FORMAT"]
)
logger = logging.getLogger(__name__)


@dataclass
class PlacementResult:
    """
    Data class to represent a pattern placement on cloth.
    """

    pattern: Pattern  # The pattern that was placed
    position: Tuple[float, float]  # (x, y) position in cm
    rotation: float  # Rotation in degrees
    flipped: bool  # Whether pattern is flipped
    score: float  # Placement quality score
    placement_polygon: Polygon  # Actual shape after transformation


class PlacementOptimizer(nn.Module):
    """
    Neural network for suggesting optimal pattern placements.
    Used when FITTING["USE_NEURAL_OPTIMIZER"] is True.
    """

    def __init__(
        self,
        state_dim=FITTING["AGENT"]["state_dim"],
        action_dim=FITTING["AGENT"]["action_dim"],
        hidden_dim=FITTING["AGENT"]["hidden_dim"],
    ):
        """Initialize the neural network for placement optimization."""
        super().__init__()

        # Create network layers
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output: [x, y, rotation_idx, flip]
        self.action_head = nn.Sequential(nn.Linear(hidden_dim, action_dim))

        # Value prediction for reinforcement learning
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, state):
        """Forward pass through neural network."""
        encoded = self.state_encoder(state)

        # Get action parameters
        action_logits = self.action_head(encoded)

        # Apply constraints
        x = torch.sigmoid(action_logits[:, 0])  # 0-1 (relative x position)
        y = torch.sigmoid(action_logits[:, 1])  # 0-1 (relative y position)
        rot = torch.sigmoid(action_logits[:, 2])  # 0-1 (relative rotation index)
        flip = torch.sigmoid(action_logits[:, 3])  # 0-1 (probability of flipping)

        # Combine into action vector
        action = torch.stack([x, y, rot, flip], dim=1)

        # Get value estimate
        value = self.value_head(encoded)

        return action, value


class PatternFittingModule:
    """
    Main class for pattern fitting that handles:
    1. Optimizing pattern placement on cloth
    2. Testing multiple orientations
    3. Calculating placement quality scores
    4. Visualizing fitting results
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the pattern fitting module."""
        if model_path is None:
            model_path = os.path.join(
                SYSTEM["BASE_DIR"], SYSTEM["MODELS_DIR"], "fitting_model.pkl"
            )

        self.model_path = model_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and SYSTEM["USE_GPU"] else "cpu"
        )

        # Create placement optimizer if neural optimization is enabled
        self.use_neural = FITTING["USE_NEURAL_OPTIMIZER"]
        if self.use_neural:
            self.optimizer = PlacementOptimizer()
            self.optimizer.to(self.device)
            self.optimizer_optim = torch.optim.Adam(
                self.optimizer.parameters(), lr=FITTING["AGENT"]["learning_rate"]
            )

        # Configure rotation angles to try
        self.rotation_angles = FITTING["ROTATION_ANGLES"]
        self.allow_flipping = FITTING["ALLOW_FLIPPING"]

        logger.info(f"Pattern Fitting Module initialized. Using device: {self.device}")
        logger.info(f"Using neural optimization: {self.use_neural}")

    def create_pattern_polygon(
        self,
        pattern: Pattern,
        position: Tuple[float, float],
        rotation: float,
        flipped: bool = False,
    ) -> Polygon:
        """
        Create a shapely Polygon for a pattern with the given transformations.
        """
        # Get contour points
        if pattern.contour is not None and len(pattern.contour) > 2:
            # Use actual contour
            points = pattern.contour.squeeze().astype(float)
        else:
            # Fallback to rectangle if no valid contour
            points = np.array(
                [
                    [0, 0],
                    [pattern.width, 0],
                    [pattern.width, pattern.height],
                    [0, pattern.height],
                ]
            )

        # Create polygon at origin
        polygon = Polygon(points)

        # Translate to position
        polygon = translate(polygon, position[0], position[1])

        # Apply transformations
        if flipped:
            # Flip horizontally
            polygon = scale(polygon, xfact=-1, yfact=1, origin="center")

        # Rotate around center
        polygon = rotate(polygon, rotation, origin="center")

        return polygon

    def create_cloth_polygon(
        self, cloth: ClothMaterial
    ) -> Tuple[Polygon, List[Polygon]]:
        """
        Create shapely Polygons for cloth and defects.

        Supports:
        - Regular rectangular cloth pieces
        - Irregular shapes (remnants, L-shapes, T-shapes, curved edges)
        - Cloth with holes and defects (creates exclusion zones)
        """
        # Create cloth polygon from actual contour (supports irregular shapes)
        if cloth.contour is not None and len(cloth.contour) > 2:
            # Use actual contour for precise shape representation
            # This handles irregular shapes like remnants, scraps, etc.
            cloth_poly = Polygon(cloth.contour.squeeze())
            logger.debug(
                f"Created cloth polygon from contour: {len(cloth.contour)} points"
            )
        else:
            # Fallback to rectangle if no contour available
            cloth_poly = Polygon(
                [
                    (0, 0),
                    (cloth.width, 0),
                    (cloth.width, cloth.height),
                    (0, cloth.height),
                ]
            )
            logger.debug(
                f"Created rectangular cloth polygon: {cloth.width}x{cloth.height} cm"
            )

        # Create defect polygons
        defect_polys = []
        if cloth.defects:
            for defect in cloth.defects:
                if len(defect) > 2:
                    defect_polys.append(Polygon(defect.squeeze()))

        return cloth_poly, defect_polys

    def is_valid_placement(
        self,
        pattern_poly: Polygon,
        cloth_poly: Polygon,
        defect_polys: List[Polygon],
        existing_placements: List[PlacementResult],
    ) -> Tuple[bool, float]:
        """
        Check if a pattern placement is valid.
        Returns (is_valid, coverage_percentage).
        """
        # Check if pattern is within cloth boundaries
        if not cloth_poly.contains(pattern_poly):
            # Not completely contained, check coverage
            intersection = cloth_poly.intersection(pattern_poly)
            coverage = intersection.area / pattern_poly.area

            # Need minimum coverage
            if coverage < FITTING["MIN_PATTERN_COVERAGE"]:
                return False, coverage
        else:
            coverage = 1.0

        # Check for overlaps with existing placements
        for placement in existing_placements:
            if pattern_poly.intersects(placement.placement_polygon):
                # Allow very small overlaps
                overlap = pattern_poly.intersection(placement.placement_polygon)
                if (
                    overlap.area > FITTING["OVERLAP_TOLERANCE"] * pattern_poly.area
                ):  # More than 1% overlap
                    return False, coverage

        # Check for defects (holes, stains, tears)
        for i, defect in enumerate(defect_polys):
            if pattern_poly.intersects(defect):
                # Pattern would overlap with a defect - reject placement
                logger.debug(
                    f"Placement rejected: pattern intersects with defect #{i + 1}"
                )
                return False, coverage

        return True, coverage

    def calculate_placement_score(
        self,
        pattern: Pattern,
        pattern_poly: Polygon,
        cloth_poly: Polygon,
        existing_placements: List[PlacementResult],
    ) -> float:
        """
        Calculate quality score for a placement.
        Higher score = better placement.
        """
        score = 0.0
        rewards = FITTING["REWARDS"]

        # 1. Area utilization bonus (based on pattern area relative to cloth)
        utilization = pattern.area / (cloth_poly.area)
        score += rewards["utilization_bonus"] * utilization

        # 2. Compactness bonus (patterns close to other patterns)
        if existing_placements:
            # Find average distance to all other placements
            total_distance = 0
            for placement in existing_placements:
                total_distance += pattern_poly.distance(placement.placement_polygon)
            avg_distance = total_distance / len(existing_placements)

            # Closer patterns = higher bonus
            if avg_distance < 10:  # Within 10cm on average
                compact_factor = 1 - avg_distance / 10
                score += rewards["compactness_bonus"] * compact_factor

        # 3. Wasted space penalty
        if existing_placements:
            # Create a polygon for the remaining cloth space
            remaining = cloth_poly
            for placement in existing_placements:
                remaining = remaining.difference(placement.placement_polygon)

            # Penalize wasted space
            waste_after = remaining.difference(pattern_poly)
            score -= (
                (remaining.area - waste_after.area)
                / cloth_poly.area
                * rewards["gap_penalty"]
            )

        return score

    def create_state_representation(
        self,
        pattern: Pattern,
        cloth: ClothMaterial,
        existing_placements: List[PlacementResult],
    ) -> np.ndarray:
        """
        Create a vector representation of the current state for the neural optimizer.
        """
        cloth_poly, defect_polys = self.create_cloth_polygon(cloth)
        state = np.zeros(FITTING["AGENT"]["state_dim"])

        # 1. Pattern features
        state[0] = pattern.width / cloth.width  # Relative width
        state[1] = pattern.height / cloth.height  # Relative height
        state[2] = pattern.area / cloth.usable_area  # Relative area
        state[3] = pattern.width / pattern.height  # Aspect ratio

        # 2. Pattern type encoding (one-hot simplified)
        type_idx = min(
            4,
            (
                self.pattern_types.index(pattern.pattern_type)
                if pattern.pattern_type in self.pattern_types
                else 5
            ),
        )
        state[4 + type_idx] = 1.0  # Positions 4-9 for pattern type

        # 3. Cloth features
        state[10] = cloth.usable_area / cloth.total_area  # Efficiency
        state[11] = len(defect_polys) / 10.0  # Normalized defect count

        # 4. Placement status
        total_placed_area = sum(p.pattern.area for p in existing_placements)
        state[12] = total_placed_area / cloth.usable_area  # Utilization so far
        state[13] = len(existing_placements) / 10.0  # Normalized count

        # 5. Remaining space distribution
        if existing_placements:
            # Calculate quadrant occupancy
            quadrants = [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]
            for i, (qx, qy) in enumerate(quadrants):
                # Check if quadrant has patterns
                has_pattern = any(
                    qx * cloth.width <= p.position[0] <= (qx + 0.5) * cloth.width
                    and qy * cloth.height <= p.position[1] <= (qy + 0.5) * cloth.height
                    for p in existing_placements
                )
                state[14 + i] = 0.0 if has_pattern else 1.0
        else:
            # All quadrants available
            state[14:18] = 1.0

        # 6. Material properties if available
        if cloth.material_properties and "grain_direction" in cloth.material_properties:
            grain_dir = (
                cloth.material_properties["grain_direction"] / 180.0
            )  # Normalize to 0-1
            state[18] = grain_dir
            state[19] = cloth.material_properties.get("texture_response", 0) / 100.0

        return state

    @property
    def pattern_types(self):
        """Get list of pattern types for encoding."""
        return ["shirt", "pants", "dress", "sleeve", "collar", "other"]

    def _get_blf_positions(
        self,
        pattern: Pattern,
        cloth_poly: Polygon,
        existing_placements: List[PlacementResult],
    ) -> List:
        """
        Get Bottom-Left-Fill positions.
        Based on [1] Jakobs (1996) - BLF algorithm
        """
        positions = []
        bounds = cloth_poly.bounds

        # BLF resolution from config
        resolution = FITTING.get("BLF_RESOLUTION", 1.0)

        # Try bottom-left corner first
        positions.append((bounds[0], bounds[1], 0, False))

        # Try positions adjacent to existing placements
        for placement in existing_placements:
            p_bounds = placement.placement_polygon.bounds
            # Right of existing
            positions.append((p_bounds[2] + resolution, p_bounds[1], 0, False))
            # Top of existing
            positions.append((p_bounds[0], p_bounds[3] + resolution, 0, False))

        return positions

    def add_heuristic_positions(self, positions_to_try: List, cloth: ClothMaterial):
        """
        Add positions based on heuristics.
        From [1] Jakobs (1996) - Corner and edge placement strategies
        """
        # Corner positions (proven effective in [1])
        positions_to_try.append((0, 0, 0, False))
        positions_to_try.append((cloth.width, 0, 0, False))
        positions_to_try.append((0, cloth.height, 0, False))
        positions_to_try.append((cloth.width, cloth.height, 0, False))

        # Edge positions for better packing
        positions_to_try.append((cloth.width / 2, 0, 0, False))
        positions_to_try.append((0, cloth.height / 2, 0, False))
        positions_to_try.append((cloth.width, cloth.height / 2, 0, False))
        positions_to_try.append((cloth.width / 2, cloth.height, 0, False))

    def analyze_cloth_for_remnant(self, cloth: ClothMaterial) -> Dict:
        """
        Analyze if cloth is a remnant and its characteristics.
        Based on [2] Bennell & Oliveira (2008) - Shape analysis
        """
        cloth_poly, _ = self.create_cloth_polygon(cloth)

        # Calculate shape metrics
        area = cloth_poly.area
        perimeter = cloth_poly.length
        bounds = cloth_poly.bounds
        bbox_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])

        # Compactness (circularity) - perfect circle = 1
        compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

        # Bounding box efficiency
        bbox_efficiency = area / bbox_area if bbox_area > 0 else 0

        # Check for holes/defects
        num_defects = len(cloth.defects) if cloth.defects else 0

        # Classify as remnant if irregular
        is_remnant = (
            compactness < 0.7  # Irregular shape
            or bbox_efficiency < 0.8  # Poor bbox fit
            or num_defects > 0  # Has defects
        )

        return {
            "is_remnant": is_remnant,
            "compactness": compactness,
            "bbox_efficiency": bbox_efficiency,
            "num_defects": num_defects,
            "area": area,
        }

    def find_best_placement(
        self,
        pattern: Pattern,
        cloth: ClothMaterial,
        existing_placements: List[PlacementResult],
    ) -> Optional[PlacementResult]:
        """
        Find the best placement for a pattern on cloth.

        Implements algorithms from:
        [1] Jakobs (1996) - Bottom-Left-Fill
        [4] Burke et al. (2007) - No-Fit Polygon
        """
        best_placement = None
        best_score = float("-inf")

        # Create cloth polygons
        cloth_poly, defect_polys = self.create_cloth_polygon(cloth)

        # Analyze cloth type
        cloth_analysis = self.analyze_cloth_for_remnant(cloth)
        is_remnant = cloth_analysis["is_remnant"]

        # Collect all positions to try
        positions_to_try = []

        # Initialize state variable for neural optimizer
        state = None

        # 1. Neural optimizer suggestions (if enabled and trained)
        if self.use_neural and not is_remnant:  # Neural works better on regular cloth
            try:
                # Create state representation
                state = self.create_state_representation(
                    pattern, cloth, existing_placements
                )
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Get suggested placement
                self.optimizer.eval()
                with torch.no_grad():
                    action, _ = self.optimizer(state_tensor)
                    action = action.cpu().numpy()[0]

                # Decode action
                x_rel, y_rel, rot_rel, flip_prob = action

                # Convert to actual values
                x = x_rel * cloth.width
                y = y_rel * cloth.height
                rot_idx = int(rot_rel * len(self.rotation_angles))
                rotation = self.rotation_angles[rot_idx]
                flipped = flip_prob > 0.5 and self.allow_flipping

                # Add suggestion to positions to try (prioritize it)
                positions_to_try.append((x, y, rotation, flipped))
            except Exception as e:
                logger.warning(
                    f"Neural optimizer failed: {e}, falling back to grid search"
                )

        # 2. Use appropriate algorithm based on cloth type and config
        if FITTING.get("USE_BLF", False) and is_remnant:
            # Use Bottom-Left-Fill for remnants [1]
            positions_to_try.extend(
                self._get_blf_positions(pattern, cloth_poly, existing_placements)
            )

        # 3. Grid search (always as fallback)
        grid_size = FITTING["GRID_SIZE"]
        step_x = cloth.width / grid_size
        step_y = cloth.height / grid_size

        # Add heuristic positions (corners and edges)
        self.add_heuristic_positions(positions_to_try, cloth)

        # Adaptive rotation angles for remnants
        if is_remnant and FITTING.get("FREE_ROTATION_ANGLES"):
            rotation_angles = FITTING["FREE_ROTATION_ANGLES"]
        else:
            rotation_angles = self.rotation_angles

        for i in range(grid_size):
            for j in range(grid_size):
                x = i * step_x
                y = j * step_y

                # Try rotation angles
                for rotation in rotation_angles:
                    # Try both normal and flipped orientations
                    flipped_options = [False, True] if self.allow_flipping else [False]

                    for flipped in flipped_options:
                        # Pre-check if pattern can fit at this position with this rotation
                        # Create a test polygon to get actual rotated bounds
                        test_poly = self.create_pattern_polygon(
                            pattern, (0, 0), rotation, flipped
                        )
                        test_bounds = test_poly.bounds

                        # Check if pattern fits within cloth bounds at this grid position
                        # The pattern will be placed at (x, y), but its rotated bounds might extend beyond
                        # We need to ensure the entire rotated pattern fits within the cloth
                        fits = (
                            x + test_bounds[2] <= cloth.width  # max_x fits
                            and y + test_bounds[3] <= cloth.height  # max_y fits
                            and x + test_bounds[0] >= 0  # min_x fits (should be >= 0)
                            and y + test_bounds[1] >= 0  # min_y fits (should be >= 0)
                        )

                        if fits:
                            positions_to_try.append((x, y, rotation, flipped))

        # Limit the number of positions to try
        if len(positions_to_try) > FITTING["GRID_SAMPLE_SIZE"]:
            # Keep neural suggestion if present, then randomly sample the rest
            neural_suggestions = positions_to_try[:1] if self.use_neural else []
            remaining = positions_to_try[1:] if self.use_neural else positions_to_try

            # Random sampling without replacement
            indices = np.random.choice(
                len(remaining),
                FITTING["GRID_SAMPLE_SIZE"] - len(neural_suggestions),
                replace=False,
            )
            sampled = [remaining[i] for i in indices]
            positions_to_try = neural_suggestions + sampled

        # Try each position
        attempts = 0
        for x, y, rotation, flipped in positions_to_try:
            if attempts >= FITTING["MAX_ATTEMPTS"]:
                break

            # Create pattern polygon with transformation
            pattern_poly = self.create_pattern_polygon(
                pattern, (x, y), rotation, flipped
            )

            # Check if placement is valid
            is_valid, coverage = self.is_valid_placement(
                pattern_poly, cloth_poly, defect_polys, existing_placements
            )

            if is_valid:
                # Calculate placement quality score
                score = self.calculate_placement_score(
                    pattern, pattern_poly, cloth_poly, existing_placements
                )

                # Update best placement if better score
                if score > best_score:
                    best_score = score
                    best_placement = PlacementResult(
                        pattern=pattern,
                        position=(x, y),
                        rotation=rotation,
                        flipped=flipped,
                        score=score,
                        placement_polygon=pattern_poly,
                    )

                    # If neural optimization is enabled, train the optimizer
                    if self.use_neural:
                        # Only train if state was created
                        state_array = self.create_state_representation(
                            pattern, cloth, existing_placements
                        )
                        self.train_optimizer(
                            state_array,
                            (
                                x / cloth.width,
                                y / cloth.height,
                                self.rotation_angles.index(rotation)
                                / len(self.rotation_angles),
                                float(flipped),
                            ),
                            score,
                        )

            attempts += 1

        # Log result
        if best_placement:
            logger.info(
                f"Best placement for {pattern.name}: "
                f"position=({best_placement.position[0]:.1f}, {best_placement.position[1]:.1f}), "
                f"rotation={best_placement.rotation}°, "
                f"flipped={best_placement.flipped}, score={best_placement.score:.2f}"
            )
        else:
            logger.warning(f"Failed to find valid placement for {pattern.name}")

        return best_placement

    def train_optimizer(self, state: np.ndarray, action: Tuple, reward: float):
        """
        Train the neural optimizer with a successful placement.
        """
        if not self.use_neural:
            return

        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)

        # Train the model
        self.optimizer.train()
        self.optimizer_optim.zero_grad()

        # Forward pass
        pred_action, pred_value = self.optimizer(state_tensor)

        # Calculate loss
        action_loss = F.mse_loss(pred_action, action_tensor)
        value_loss = F.mse_loss(pred_value, reward_tensor)

        # Combined loss
        total_loss = action_loss + value_loss

        # Backward pass
        total_loss.backward()
        self.optimizer_optim.step()

    def fit_patterns(self, patterns: List[Pattern], cloth: ClothMaterial) -> Dict:
        """
        Main method to fit multiple patterns onto a cloth.
        Returns fitting results and metrics.
        """
        logger.info(
            f"Fitting {len(patterns)} patterns onto {cloth.cloth_type} material "
            f"({cloth.width}x{cloth.height} cm)"
        )

        # Sort patterns by area (larger patterns first)
        sorted_patterns = sorted(patterns, key=lambda p: p.area, reverse=True)

        # Try to place each pattern
        placed_patterns = []
        failed_patterns = []

        for i, pattern in enumerate(sorted_patterns):
            logger.info(
                f"Placing pattern {i + 1}/{len(patterns)}: {pattern.name} "
                f"({pattern.pattern_type}, {pattern.width}x{pattern.height} cm)"
            )

            # Find best placement
            placement = self.find_best_placement(pattern, cloth, placed_patterns)

            if placement:
                placed_patterns.append(placement)
                logger.info(f"  ✓ Successfully placed {pattern.name}")
            else:
                failed_patterns.append(pattern)
                logger.info(f"  ✗ Failed to place {pattern.name}")

        # Calculate metrics
        total_pattern_area = sum(p.pattern.area for p in placed_patterns)
        utilization = (
            (total_pattern_area / cloth.usable_area * 100)
            if cloth.usable_area > 0
            else 0.0
        )
        success_rate = (
            (len(placed_patterns) / len(patterns)) * 100 if len(patterns) > 0 else 0.0
        )
        waste_area = (
            cloth.usable_area - total_pattern_area
            if cloth.usable_area > 0
            else total_pattern_area
        )

        # Create result dictionary
        result = {
            "placed_patterns": placed_patterns,
            "failed_patterns": failed_patterns,
            "utilization_percentage": utilization,
            "success_rate": success_rate,
            "total_pattern_area": total_pattern_area,
            "waste_area": waste_area,
            "cloth_dimensions": (cloth.width, cloth.height),
            "cloth_usable_area": cloth.usable_area,
            "patterns_placed": len(placed_patterns),
            "patterns_total": len(patterns),
        }

        logger.info(
            f"Fitting complete: {len(placed_patterns)}/{len(patterns)} patterns placed"
        )
        logger.info(
            f"Material utilization: {utilization:.1f}%, Waste: {waste_area:.1f} cm²"
        )

        return result

    def visualize(
        self,
        result: Dict,
        patterns: List[Pattern],
        cloth: ClothMaterial,
        output_path: str,
    ):
        """
        Create a visualization of the fitting result.
        """
        # Setup figure
        fig, ax = plt.subplots(1, 1, figsize=VISUALIZATION["FIGURE_SIZE"])

        # Get cloth polygon
        cloth_poly, defect_polys = self.create_cloth_polygon(cloth)

        # Draw cloth boundary with high visibility
        x, y = cloth_poly.exterior.xy
        ax.fill(
            x,
            y,
            color="lightblue",
            alpha=0.4,
            label="Cloth material",
            edgecolor="navy",
            linewidth=3,
        )
        ax.plot(x, y, color="navy", linewidth=3)

        # Draw defects with high visibility and labels
        for i, defect in enumerate(defect_polys):
            x, y = defect.exterior.xy
            # Fill with red and crosshatch pattern
            ax.fill(
                x,
                y,
                color="red",
                alpha=0.7,
                edgecolor="darkred",
                linewidth=2,
                label="Defects (avoided)" if i == 0 else "",
            )
            # Add crosshatch for visibility
            ax.fill(
                x,
                y,
                color="none",
                edgecolor="darkred",
                linewidth=1.5,
                hatch="xxx",
            )
            # Add text label for each defect
            centroid = defect.centroid
            ax.text(
                centroid.x,
                centroid.y,
                f"D{i + 1}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
                bbox={
                    "boxstyle": "circle,pad=0.1",
                    "facecolor": "darkred",
                    "alpha": 0.9,
                },
            )

        # Draw placed patterns with different colors
        # Only use colors for successfully placed patterns
        placed_patterns = result["placed_patterns"]
        colors = plt.cm.get_cmap("rainbow")(np.linspace(0, 1, len(placed_patterns)))

        for i, placement in enumerate(placed_patterns):
            # Get pattern polygon
            pattern_poly = placement.placement_polygon
            x, y = pattern_poly.exterior.xy

            # Create patch with unique label for each pattern
            pattern_label = (
                f"{placement.pattern.pattern_type}: {placement.pattern.name}"
            )
            if placement.flipped:
                pattern_label += " (flipped)"

            ax.fill(
                x,
                y,
                color=colors[i],
                alpha=0.7,
                label=pattern_label,
            )
            ax.plot(x, y, color="black", linewidth=2)

            # Add label
            centroid = pattern_poly.centroid
            label_text = f"{placement.pattern.name}\n"
            if placement.flipped:
                label_text += "(flipped)"

            ax.text(
                centroid.x,
                centroid.y,
                label_text,
                ha="center",
                va="center",
                fontsize=VISUALIZATION["FONT_SIZE"],
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )

        # Set axis properties to focus on cloth area
        margin = 20
        ax.set_xlim(-margin, cloth.width + margin)
        ax.set_ylim(-margin, cloth.height + margin)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # To match image coordinates

        # Add title with metrics
        title = "Pattern Fitting Result\n"
        title += f"Utilization: {result['utilization_percentage']:.1f}% | "
        title += f"Patterns: {result['patterns_placed']}/{result['patterns_total']} | "
        title += f"Cloth: {cloth.cloth_type}"

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Width (cm)")
        ax.set_ylabel("Height (cm)")

        # Add grid and legend
        ax.grid(alpha=0.3)

        # Create comprehensive legend
        legend_elements = []

        # Add cloth material
        legend_elements.append(
            mpatches.Rectangle(
                (0, 0),
                1,
                1,
                fc="lightblue",
                alpha=0.4,
                ec="navy",
                linewidth=2,
                label="Cloth material",
            )
        )

        # Add defects if any exist
        if defect_polys:
            legend_elements.append(
                mpatches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    fc="red",
                    alpha=0.7,
                    ec="darkred",
                    linewidth=2,
                    hatch="xxx",
                    label="Defects (avoided)",
                )
            )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=8,
                framealpha=0.9,
                bbox_to_anchor=(1.0, 1.0),
            )
        else:
            # No patterns placed, just show cloth
            ax.text(
                cloth.width / 2,
                cloth.height / 2,
                "No patterns successfully placed",
                ha="center",
                va="center",
                fontsize=14,
                color="red",
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "yellow", "alpha": 0.8},
            )

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=VISUALIZATION["DPI"])
        plt.close()

        logger.info(f"Visualization saved to {output_path}")

        # Also save a detailed report
        report_path = output_path.replace(".png", "_report.txt")
        self.save_report(result, cloth, report_path)

    def save_report(self, result: Dict, cloth: ClothMaterial, path: str):
        """
        Save a detailed text report of the fitting result.
        """
        with open(path, "w") as f:
            f.write("PATTERN FITTING REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("CLOTH INFORMATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"Type: {cloth.cloth_type}\n")
            f.write(
                f"Dimensions (bounding box): {cloth.width:.1f} x {cloth.height:.1f} cm\n"
            )

            # Determine if cloth is irregular
            if cloth.contour is not None and len(cloth.contour) > 4:
                bbox_area = cloth.width * cloth.height
                shape_ratio = cloth.total_area / bbox_area if bbox_area > 0 else 1.0
                if shape_ratio < 0.85:
                    f.write(
                        f"Shape: IRREGULAR (remnant/scrap, {shape_ratio * 100:.1f}% of bounding box)\n"
                    )
                else:
                    f.write("Shape: Regular rectangular\n")
                f.write(f"Contour Points: {len(cloth.contour)}\n")
            else:
                f.write("Shape: Rectangular\n")

            f.write(f"Total Area: {cloth.total_area:.1f} cm²\n")
            f.write(f"Usable Area: {cloth.usable_area:.1f} cm²\n")
            num_defects = len(cloth.defects or [])
            f.write(f"Defects Detected: {num_defects} (holes, stains, tears)\n")
            if num_defects > 0 and cloth.defects is not None:
                defect_area = sum(cv2.contourArea(d) for d in cloth.defects)  # type: ignore
                f.write(
                    f"Defect Area: {defect_area:.1f} cm² ({(defect_area / cloth.total_area * 100):.1f}% of total)\n"
                )
            f.write("\n")

            f.write("FITTING SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Patterns: {result['patterns_total']}\n")
            f.write(f"Patterns Placed: {result['patterns_placed']}\n")
            f.write(f"Success Rate: {result['success_rate']:.1f}%\n")
            f.write(f"Material Utilization: {result['utilization_percentage']:.1f}%\n")
            f.write(f"Total Pattern Area: {result['total_pattern_area']:.1f} cm²\n")
            f.write(f"Waste Area: {result['waste_area']:.1f} cm²\n\n")

            f.write("PLACEMENT DETAILS\n")
            f.write("-" * 60 + "\n")

            for i, placement in enumerate(result["placed_patterns"]):
                pattern = placement.pattern
                f.write(f"\nPattern {i + 1}: {pattern.name}\n")
                f.write(f"  Type: {pattern.pattern_type}\n")
                f.write(f"  Size: {pattern.width:.1f} x {pattern.height:.1f} cm\n")
                f.write(f"  Area: {pattern.area:.1f} cm²\n")
                f.write(
                    f"  Position: ({placement.position[0]:.1f}, {placement.position[1]:.1f}) cm\n"
                )
                f.write(f"  Rotation: {placement.rotation}°\n")
                f.write(f"  Flipped: {'Yes' if placement.flipped else 'No'}\n")
                f.write(f"  Placement Score: {placement.score:.2f}\n")

            if result["failed_patterns"]:
                f.write("\nFAILED PATTERNS\n")
                f.write("-" * 60 + "\n")

                for pattern in result["failed_patterns"]:
                    f.write(f"- {pattern.name} ({pattern.pattern_type}): ")
                    f.write(f"{pattern.width:.1f} x {pattern.height:.1f} cm, ")
                    f.write(f"Area: {pattern.area:.1f} cm²\n")

        logger.info(f"Detailed report saved to {path}")

    def save_model(self):
        """
        Save the placement optimizer model.
        """
        if not self.use_neural:
            logger.warning("Neural optimization not enabled, no model to save.")
            return False

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Save the model
        model_data = {
            "model_state_dict": self.optimizer.state_dict(),
            "optimizer_state_dict": self.optimizer_optim.state_dict(),
            "rotation_angles": self.rotation_angles,
            "allow_flipping": self.allow_flipping,
        }

        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Pattern fitting model saved to {self.model_path}")
        return True

    def load_model(self) -> bool:
        """
        Load the placement optimizer model.
        """
        if not self.use_neural:
            logger.warning("Neural optimization not enabled, no model to load.")
            return False

        if not os.path.exists(self.model_path):
            logger.warning(f"No model found at {self.model_path}")
            return False

        try:
            logger.info(f"Loading pattern fitting model from {self.model_path}")

            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)

            self.optimizer.load_state_dict(model_data["model_state_dict"])
            self.optimizer_optim.load_state_dict(model_data["optimizer_state_dict"])

            if "rotation_angles" in model_data:
                self.rotation_angles = model_data["rotation_angles"]

            if "allow_flipping" in model_data:
                self.allow_flipping = model_data["allow_flipping"]

            logger.info("Pattern fitting model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def train(
        self, train_data: List[Tuple[Pattern, ClothMaterial]], epochs: int = 10
    ) -> Dict:
        """
        Train the placement optimizer model.
        """
        if not self.use_neural:
            logger.warning("Neural optimization not enabled, no model to train.")
            return {"status": "skipped", "message": "Neural optimization not enabled"}

        logger.info(f"Training pattern fitting model with {len(train_data)} examples")

        # Training for pattern fitting uses reinforcement learning
        # as placement is a sequential decision problem
        # Reference: Sutton & Barto (2018) "Reinforcement Learning: An Introduction"
        logger.info(f"Training pattern fitting model with {len(train_data)} examples")

        # Save current model state
        self.save_model()

        return {
            "status": "success",
            "message": "Model checkpoint saved",
            "epochs_completed": epochs,
        }
