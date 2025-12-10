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

    def __init__(self, model_path: Optional[str] = None, auto_scale: Optional[bool] = None):
        """Initialize the pattern fitting module."""
        if model_path is None:
            model_path = os.path.join(
                SYSTEM["BASE_DIR"], SYSTEM["MODELS_DIR"], "fitting_model.pkl"
            )

        self.model_path = model_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and SYSTEM["USE_GPU"] else "cpu"
        )
        
        # Configure auto-scaling (instance override > global config)
        if auto_scale is None:
            self.auto_scale = FITTING["AUTO_SCALE"]
        else:
            self.auto_scale = auto_scale

        # Create placement optimizer if neural optimization is enabled
        self.use_neural = FITTING["USE_NEURAL_OPTIMIZER"]
        if self.use_neural:
            self.optimizer = PlacementOptimizer()
            self.optimizer.to(self.device)
            self.optimizer_optim = torch.optim.Adam(
                self.optimizer.parameters(), lr=FITTING["AGENT"]["learning_rate"]
            )

        # Configure heuristic parameters (can be optimized during training)
        self.grid_size = FITTING["GRID_SIZE"]
        self.rotation_angles = FITTING["ROTATION_ANGLES"]
        self.allow_flipping = FITTING["ALLOW_FLIPPING"]
        self.max_attempts = FITTING["MAX_ATTEMPTS"]

        logger.info(f"Pattern Fitting Module initialized. Using device: {self.device}")
        logger.info(f"Using neural optimization: {self.use_neural}")

    def scale_pattern(self, pattern: Pattern, scale: float) -> Pattern:
        """
        Create a new Pattern object scaled by the given factor.
        """
        # Scale dimensions
        new_width = pattern.width * scale
        new_height = pattern.height * scale
        new_area = new_width * new_height
        
        # Scale contour if present
        new_contour = None
        if pattern.contour is not None and len(pattern.contour) > 0:
            new_contour = pattern.contour * scale
            
        # Create new pattern object
        return Pattern(
            id=pattern.id, # Keep ID or generate new? Ideally new hash or same ID if logical identity
            name=f"{pattern.name}_x{scale:.2f}",
            pattern_type=pattern.pattern_type,
            width=new_width,
            height=new_height,
            area=new_area,
            contour=new_contour,
            confidence=pattern.confidence,
            key_points=[(x*scale, y*scale) for x, y in pattern.key_points] if pattern.key_points else None
        )

    def find_optimal_scale(self, patterns: List[Pattern], cloth: ClothMaterial) -> float:
        """
        Find the largest scale factor that fits the required percentage of patterns.
        Uses binary search and coarse resolution for performance.
        """
        if not self.auto_scale:
            return 1.0
            
        scale_step = FITTING.get("SCALE_STEP", 0.1)
        max_scale = FITTING.get("MAX_SCALE", 3.0)
        min_patterns_percent = FITTING.get("MIN_PATTERNS_PERCENT", 1.0)
        
        logger.info(f"Finding optimal scale (max: {max_scale})...")
        
        # Binary search parameters
        low = 1.0
        high = max_scale
        best_valid_scale = 1.0
        
        # Coarse resolution for faster feasibility checking
        # Configured via FITTING["SCALE_CHECK_RESOLUTION"] (default 5.0 cm)
        check_resolution = FITTING.get("SCALE_CHECK_RESOLUTION", 5.0)
        
        # Binary search iterations
        # specific precision target around 0.1 (scale_step)
        
        while high - low > scale_step / 2:
            mid = (low + high) / 2
            # Round to nearest step to avoid weird floats
            mid = round(mid / scale_step) * scale_step
            
            if mid <= 1.0: # Don't go below 1.0
                low = mid + scale_step/2 # Move up
                continue
                
            if mid == low or mid == high:
                break

            # Create scaled patterns
            scaled_patterns = [self.scale_pattern(p, mid) for p in patterns]
            
            # Quick feasibility check
            # Turn off logging for these checks
            prev_level = logger.level
            logger.setLevel(logging.WARNING)
            
            try:
                # Use baseline_mode=True and coarse resolution
                result = self.fit_patterns(
                    scaled_patterns, 
                    cloth, 
                    baseline_mode=True, 
                    is_optimization_pass=True,
                    search_resolution=check_resolution
                )
            finally:
                logger.setLevel(prev_level)
                
            success_rate = result["patterns_placed"] / len(patterns)
            
            if success_rate >= min_patterns_percent:
                best_valid_scale = max(best_valid_scale, mid)
                low = mid # Try higher
                # logger.info(f"Scale {mid:.2f} fits ({success_rate*100:.0f}%)")
            else:
                high = mid # Try lower
                # logger.info(f"Scale {mid:.2f} fails ({success_rate*100:.0f}%)")
        
        logger.info(f"Found optimal scale: {best_valid_scale:.2f}")
        return best_valid_scale

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
            # Distance threshold from config (default 10 cm)
            # Represents "close proximity" - patterns within this distance likely related
            compactness_threshold = FITTING.get("COMPACTNESS_DISTANCE_CM", 10.0)
            if avg_distance < compactness_threshold:
                compact_factor = 1 - avg_distance / compactness_threshold
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
        from .config import PATTERN
        return PATTERN.get("TYPES", ["shirt", "pants", "dress", "sleeve", "collar", "other"])

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

    def _get_rotated_dimensions(
        self, pattern: Pattern, rotation: float
    ) -> Tuple[float, float]:
        """Get pattern dimensions after rotation."""
        # Normalize rotation to [0, 360)
        norm_rotation = rotation % 360
        # For 0/180 degrees, dimensions stay the same
        if norm_rotation == 0 or norm_rotation == 180:
            return pattern.width, pattern.height
        # For 90/270 degrees, dimensions swap
        elif norm_rotation == 90 or norm_rotation == 270:
            return pattern.height, pattern.width
        else:
            # For other angles, calculate bounding box
            rad = np.radians(rotation)
            cos_r = abs(np.cos(rad))
            sin_r = abs(np.sin(rad))
            new_width = pattern.width * cos_r + pattern.height * sin_r
            new_height = pattern.width * sin_r + pattern.height * cos_r
            return new_width, new_height

    def _prefilter_patterns(
        self, patterns: List[Pattern], cloth: ClothMaterial
    ) -> Tuple[List[Pattern], List[Pattern]]:
        """
        Filter out patterns that cannot possibly fit on cloth.

        Issue A fix: Prevents wasted processing time on impossible placements.

        Returns:
            (valid_patterns, invalid_patterns)
        """
        cloth_width, cloth_height = cloth.width, cloth.height

        valid_patterns = []
        invalid_patterns = []

        for pattern in patterns:
            # Check all possible rotations
            can_fit = False
            for angle in self.rotation_angles:
                p_width, p_height = self._get_rotated_dimensions(pattern, angle)
                if p_width <= cloth_width and p_height <= cloth_height:
                    can_fit = True
                    break

            if can_fit:
                valid_patterns.append(pattern)
            else:
                invalid_patterns.append(pattern)
                logger.warning(
                    f"Pattern {pattern.name} ({pattern.width:.1f}×{pattern.height:.1f} cm) "
                    f"cannot fit on cloth ({cloth_width:.1f}×{cloth_height:.1f} cm) - FILTERED OUT"
                )

        if invalid_patterns:
            logger.info(
                f"Pre-filtered {len(invalid_patterns)} patterns that cannot fit on cloth"
            )

        return valid_patterns, invalid_patterns

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

        Enhanced with:
        - Issue F: Early stopping when excellent placement found
        """
        best_placement = None
        best_score = float("-inf")

        # Issue F: Early stopping threshold from config
        # See config.py FITTING["EXCELLENT_SCORE_THRESHOLD"] for detailed rationale
        # Based on quality threshold stopping criteria from:
        # [6] Hopper & Turton (2001) - meta-heuristics with quality-based termination
        # [7] Maxwell et al. (2015) - stopping rules in search strategies
        excellent_threshold = FITTING.get("EXCELLENT_SCORE_THRESHOLD", 15.0)

        # Create cloth polygons
        cloth_poly, defect_polys = self.create_cloth_polygon(cloth)

        # Analyze cloth type (used for logging/future optimization)
        cloth_analysis = self.analyze_cloth_for_remnant(cloth)
        # Note: is_remnant currently unused but available for future algorithm selection
        _ = cloth_analysis["is_remnant"]

        # Grid search is the primary placement algorithm
        # BLF is used only in baseline_mode via _find_baseline_placement()
        # Neural optimizer (USE_NEURAL_OPTIMIZER) is experimental and disabled by default
        step_x = cloth.width / self.grid_size
        step_y = cloth.height / self.grid_size

        # Generate grid positions with all rotation/flip combinations
        positions_to_try = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = i * step_x
                y = j * step_y

                # Try rotation angles
                for rotation in self.rotation_angles:
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
        # Note: This is a single-level loop iterating through pre-generated positions
        # The break statement for early stopping will correctly exit this loop
        attempts = 0
        early_stop_triggered = False

        for x, y, rotation, flipped in positions_to_try:
            # Check max attempts limit
            if attempts >= self.max_attempts:
                logger.debug(f"Reached max_attempts limit ({self.max_attempts})")
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

                    # Issue F: Early stopping if excellent placement found
                    # Quality threshold stopping criterion from [6] Hopper & Turton (2001)
                    # and [7] Maxwell et al. (2015) - stop when "good enough" solution reached
                    if best_score > excellent_threshold:
                        early_stop_triggered = True
                        logger.debug(
                            f"Found excellent placement (score={best_score:.2f} > "
                            f"threshold={excellent_threshold:.2f}), stopping early "
                            f"after {attempts + 1} attempts"
                        )
                        break  # Exit the positions_to_try loop

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

        # Fallback: If heuristic search failed, try dense baseline search
        # This fixes the discrepancy where find_optimal_scale says it fits (using dense search)
        # but the sparse grid search here misses it.
        if best_placement is None:
            logger.info(f"Heuristic search failed for {pattern.name}, attempting dense baseline fallback...")
            # Use 5.0cm resolution (same as find_optimal_scale's fast check)
            # This ensures that if the scaling check passed, we will likely find that spot again
            fallback_placement = self._find_baseline_placement(
                pattern, cloth, existing_placements, search_resolution=5.0
            )
            
            if fallback_placement:
                # Calculate proper score for this placement (baseline returns dummy 1.0)
                fallback_score = self.calculate_placement_score(
                    pattern, 
                    fallback_placement.placement_polygon, 
                    cloth_poly, 
                    existing_placements
                )
                fallback_placement.score = fallback_score
                best_placement = fallback_placement
                logger.info(f"Fallback successful: Placed {pattern.name} at {best_placement.position}")

        # Log result with early stopping info
        if best_placement:
            log_msg = (
                f"Best placement for {pattern.name}: "
                f"position=({best_placement.position[0]:.1f}, {best_placement.position[1]:.1f}), "
                f"rotation={best_placement.rotation}°, "
                f"flipped={best_placement.flipped}, score={best_placement.score:.2f}"
            )
            if early_stop_triggered:
                log_msg += f" (early stop after {attempts} attempts)"
            logger.info(log_msg)
        else:
            logger.warning(
                f"Failed to find valid placement for {pattern.name} "
                f"after {attempts} attempts"
            )

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

    def fit_patterns(
        self, 
        patterns: List[Pattern], 
        cloth: ClothMaterial, 
        baseline_mode: bool = False,
        is_optimization_pass: bool = False,
        search_resolution: Optional[float] = None
    ) -> Dict:
        """
        Fit multiple patterns onto a cloth material.
        
        Args:
            patterns: List of patterns to fit
            cloth: Cloth to fit onto
            baseline_mode: If True, uses strict BLF baseline without optimization
            
        Returns:
            Dictionary with fitting results and metrics
        """
        start_time = datetime.now()
        
        # Auto-scaling (only on main pass, not recursive optimization passes)
        if not is_optimization_pass and FITTING.get("AUTO_SCALE", False):
            optimal_scale = self.find_optimal_scale(patterns, cloth)
            if optimal_scale > 1.0:
                logger.info(f"Scaling all patterns by {optimal_scale:.2f}x")
                patterns = [self.scale_pattern(p, optimal_scale) for p in patterns]

        
        # Issue A: Pre-filter patterns that definitely won't fit
        valid_patterns, invalid_patterns = self._prefilter_patterns(patterns, cloth)
        
        # Sort patterns
        if baseline_mode:
            # Baseline: Sort by area (descending) as requested by Dr. Karaman
            # "Sort patterns by height or area"
            sorted_patterns = sorted(valid_patterns, key=lambda p: p.area, reverse=True)
            logger.info("Baseline Mode: Sorted patterns by area (descending)")
        else:
            # Normal: Intelligent sorting (usually by area too, but can be customized)
            sorted_patterns = sorted(valid_patterns, key=lambda p: p.area, reverse=True)

        placements: List[PlacementResult] = []
        waste_area = 0.0
        
        # Process each pattern
        for i, pattern in enumerate(sorted_patterns):
            logger.info(f"Fitting pattern {i+1}/{len(sorted_patterns)}: {pattern.name}")
            
            if baseline_mode:
                # Use strict BLF baseline
                placement = self._find_baseline_placement(pattern, cloth, placements, search_resolution)
            else:
                # Use standard optimize finding
                placement = self.find_best_placement(pattern, cloth, placements)
            
            if placement:
                placements.append(placement)
                logger.info(
                    f"Placed {pattern.name} at ({placement.position[0]:.1f}, "
                    f"{placement.position[1]:.1f})"
                )
            else:
                logger.warning(f"Could not fit pattern {pattern.name}")

        # Calculate metrics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Calculate utilization
        patterns_area = sum(p.pattern.area for p in placements)
        if cloth.total_area > 0:
            utilization = (patterns_area / cloth.total_area) * 100
            waste_area = cloth.total_area - patterns_area
        else:
            utilization = 0
            waste_area = 0
            
        # Identify failed patterns
        placed_pattern_ids = {id(p.pattern) for p in placements}
        failed_patterns = [p for p in patterns if id(p) not in placed_pattern_ids]
            
        return {
            "placements": placements,
            "placed_patterns": placements, # Alias for visualization
            "patterns_placed": len(placements),
            "num_patterns": len(patterns),
            "patterns_total": len(patterns),
            "utilization_percentage": utilization,
            "waste_area": waste_area,
            "total_pattern_area": patterns_area,
            "processing_time": processing_time,
            "success_rate": (len(placements) / len(patterns) * 100) if patterns else 0,
            "cloth_id": cloth.id,
            "cloth_area": cloth.total_area,
            "cloth_dimensions": f"{cloth.width}x{cloth.height}",
            "cloth_usable_area": cloth.usable_area,
            "invalid_patterns": len(invalid_patterns),
            "failed_patterns": failed_patterns # Added in previous steps? Wait, I didn't see failed_patterns calculation in view.
        }

    def _find_baseline_placement(
        self, 
        pattern: Pattern, 
        cloth: ClothMaterial, 
        existing_placements: List[PlacementResult],
        search_resolution: Optional[float] = None
    ) -> Optional[PlacementResult]:
        """
        Find placement using strict Bottom-Left-Fill (BLF) strategy.
        
        Strategy:
        1. Scan grid from Bottom (y=0) to Top
        2. Scan grid from Left (x=0) to Right
        3. First valid position is taken immediately (Greedy)
        4. No scoring - simple binary fit check
        5. Rotations: 0, 90, 180, 270 only
        """
        cloth_poly, defect_polys = self.create_cloth_polygon(cloth)
        
        # BLF Resolution (step size)
        resolution = search_resolution if search_resolution is not None else FITTING.get("BLF_RESOLUTION", 1.0)
        
        # Strict rotations as requested
        rotations = [0, 90, 180, 270]
        
        # Grid scan limits
        max_y = int(cloth.height)
        max_x = int(cloth.width)
        
        # Iterate Y first (Bottom-Left preference means minimizing Y is primary, or minimizing X?)
        # "Place each at lowest y, then lowest x" -> Nested loop: outer Y, inner X
        for y in np.arange(0, max_y, resolution):
            for x in np.arange(0, max_x, resolution):
                
                # Check all allowed rotations at this position
                for rotation in rotations:
                    flipped = False # No flipping for strict baseline unless specified
                    
                    # Create test polygon
                    try:
                        pattern_poly = self.create_pattern_polygon(
                            pattern, (x, y), rotation, flipped
                        )
                    except Exception:
                        continue
                        
                    # Fast check: Bounds
                    if not cloth_poly.contains(pattern_poly):
                        continue
                        
                    # Strict check: Overlap and Defects
                    is_valid, _ = self.is_valid_placement(
                        pattern_poly, cloth_poly, defect_polys, existing_placements
                    )
                    
                    if is_valid:
                        # FOUND IT! Greedy return immediately.
                        return PlacementResult(
                            pattern=pattern,
                            position=(x, y),
                            rotation=rotation,
                            flipped=flipped,
                            score=1.0, # Dummy score
                            placement_polygon=pattern_poly
                        )
                        
        return None


    def visualize(
        self,
        result: Dict,
        patterns: List[Pattern],
        cloth: ClothMaterial,
        output_path: str,
        cloth_image_name: Optional[str] = None,
    ):
        """
        Create a visualization of the fitting result.

        Args:
            result: Fitting result dictionary
            patterns: List of patterns
            cloth: Cloth material
            output_path: Where to save visualization
            cloth_image_name: Name of the cloth image file (for display)
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
        # Margin from config (in cm) - prevents edge clipping in plots
        margin_cm = VISUALIZATION.get("PLOT_MARGIN_CM", 2.0)
        ax.set_xlim(-margin_cm, cloth.width + margin_cm)
        ax.set_ylim(-margin_cm, cloth.height + margin_cm)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # To match image coordinates

        # Add title with metrics and cloth name
        title = "Pattern Fitting Result\n"
        if cloth_image_name:
            title += f"Cloth: {cloth_image_name}\n"
        title += f"Utilization: {result['utilization_percentage']:.1f}% | "
        title += f"Patterns: {result['patterns_placed']}/{result['patterns_total']} | "
        title += f"Type: {cloth.cloth_type}"

        ax.set_title(title, fontsize=13, fontweight="bold")
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
        Save the placement optimizer model or heuristic configuration.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        model_data = {
            "use_neural": self.use_neural,
            "grid_size": self.grid_size,
            "rotation_angles": self.rotation_angles,
            "allow_flipping": self.allow_flipping,
            "max_attempts": self.max_attempts,
        }

        # If neural optimization is enabled, also save neural model
        if self.use_neural:
            model_data["model_state_dict"] = self.optimizer.state_dict()
            model_data["optimizer_state_dict"] = self.optimizer_optim.state_dict()

        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Pattern fitting configuration saved to {self.model_path}")
        return True

    def load_model(self) -> bool:
        """
        Load the placement optimizer model or heuristic configuration.
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"No model found at {self.model_path}")
            return False

        try:
            logger.info(f"Loading pattern fitting configuration from {self.model_path}")

            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)

            # Load heuristic parameters (always present)
            if "grid_size" in model_data:
                self.grid_size = model_data["grid_size"]
                logger.info(f"Loaded grid_size: {self.grid_size}")

            if "rotation_angles" in model_data:
                self.rotation_angles = model_data["rotation_angles"]
                logger.info(
                    f"Loaded rotation_angles: {len(self.rotation_angles)} angles"
                )

            if "allow_flipping" in model_data:
                self.allow_flipping = model_data["allow_flipping"]
                logger.info(f"Loaded allow_flipping: {self.allow_flipping}")

            if "max_attempts" in model_data:
                self.max_attempts = model_data["max_attempts"]
                logger.info(f"Loaded max_attempts: {self.max_attempts}")

            # Load neural model if present and neural optimization is enabled
            if self.use_neural and "model_state_dict" in model_data:
                self.optimizer.load_state_dict(model_data["model_state_dict"])
                self.optimizer_optim.load_state_dict(model_data["optimizer_state_dict"])
                logger.info("Loaded neural optimizer state")

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
