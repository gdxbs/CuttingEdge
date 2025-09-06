"""
Enhanced Pattern Fitting Module
Optimizes placement of fixed pattern shapes onto variable cloth materials.
Tries multiple orientations (rotation, flip) to maximize material utilization.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MPLPolygon
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, scale
import pickle
from datetime import datetime

from .enhanced_pattern_recognition import Pattern
from .enhanced_cloth_recognition import ClothMaterial
from .simple_config import FITTING, SYSTEM, VISUALIZATION

# Setup logging
logging.basicConfig(level=logging.INFO, format=SYSTEM["LOG_FORMAT"])
logger = logging.getLogger(__name__)


@dataclass
class PlacedPattern:
    """Represents a pattern that has been placed on cloth"""

    pattern: Pattern
    position: Tuple[float, float]  # (x, y) in cm from top-left
    rotation: float  # Rotation angle in degrees
    flipped: bool  # Whether pattern is flipped
    placement_score: float  # Quality score of this placement
    actual_polygon: Polygon  # Actual shape after transformation


class PlacementOptimizer(nn.Module):
    """
    Neural network that learns optimal pattern placement strategies.
    Considers cloth shape, existing patterns, and multiple orientations.
    """

    def __init__(self, state_dim: int = 20, action_dim: int = 4):
        super().__init__()

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )

        # Action predictor
        # Output: [x_normalized, y_normalized, rotation_idx, flip]
        self.action_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_dim)
        )

        # Value predictor (for reinforcement learning)
        self.value_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, state):
        encoded = self.state_encoder(state)
        action = self.action_head(encoded)
        value = self.value_head(encoded)

        # Apply constraints to action
        action_constrained = torch.zeros_like(action)
        action_constrained[:, 0] = torch.sigmoid(action[:, 0])  # x: 0-1
        action_constrained[:, 1] = torch.sigmoid(action[:, 1])  # y: 0-1
        action_constrained[:, 2] = torch.sigmoid(action[:, 2])  # rotation: 0-1
        action_constrained[:, 3] = torch.sigmoid(action[:, 3])  # flip: 0-1

        return action_constrained, value


class EnhancedPatternFitter:
    """
    Advanced pattern fitting system that optimally places patterns on cloth.
    Handles irregular cloth shapes, multiple orientations, and defects.
    """

    def __init__(self, model_path: str = "models/placement_optimizer.pkl"):
        self.model_path = model_path
        self.optimizer = PlacementOptimizer()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and SYSTEM["USE_GPU"] else "cpu"
        )
        self.optimizer.to(self.device)

        # Rotation angles to try
        self.rotation_angles = FITTING["ROTATION_ANGLES"]

        # Training components
        self.optim = torch.optim.Adam(
            self.optimizer.parameters(), lr=FITTING["AGENT"]["learning_rate"]
        )

        logger.info(f"Enhanced Pattern Fitter initialized. Device: {self.device}")

    def create_pattern_polygon(
        self, pattern: Pattern, x: float, y: float, rotation: float, flip: bool
    ) -> Polygon:
        """
        Create a Shapely polygon for the pattern at given position and orientation.
        """
        # Start with pattern contour (assume it's centered at origin)
        if len(pattern.contour) > 0:
            points = pattern.contour.squeeze().astype(float)
        else:
            # Fallback to rectangle if no contour
            points = np.array(
                [
                    [0, 0],
                    [pattern.width, 0],
                    [pattern.width, pattern.height],
                    [0, pattern.height],
                ]
            )

        # Create polygon
        poly = Polygon(points)

        # Apply transformations
        if flip:
            poly = scale(poly, xfact=-1, yfact=1, origin="center")

        poly = rotate(poly, rotation, origin="center")

        # Translate to position
        from shapely.affinity import translate

        poly = translate(poly, xoff=x, yoff=y)

        return poly

    def check_placement_validity(
        self,
        pattern_poly: Polygon,
        cloth: ClothMaterial,
        placed_patterns: List[PlacedPattern],
    ) -> Tuple[bool, float]:
        """
        Check if a pattern placement is valid and calculate coverage.
        Returns (is_valid, coverage_percentage)
        """
        # Create cloth polygon
        cloth_poly = Polygon(cloth.contour.squeeze())

        # Check if pattern is within cloth boundaries
        if not cloth_poly.contains(pattern_poly):
            intersection = cloth_poly.intersection(pattern_poly)
            coverage = intersection.area / pattern_poly.area

            # Need at least 95% coverage
            if coverage < FITTING["MIN_PATTERN_COVERAGE"]:
                return False, coverage
        else:
            coverage = 1.0

        # Check for overlaps with existing patterns
        for placed in placed_patterns:
            if pattern_poly.intersects(placed.actual_polygon):
                # Small overlaps might be OK (e.g., shared cutting lines)
                overlap = pattern_poly.intersection(placed.actual_polygon)
                if overlap.area > 0.01 * pattern_poly.area:  # More than 1% overlap
                    return False, coverage

        # Check for defects in cloth
        for defect_contour in cloth.defects:
            if len(defect_contour) > 2:
                defect_poly = Polygon(defect_contour.squeeze())
                if pattern_poly.intersects(defect_poly):
                    return False, coverage

        return True, coverage

    def calculate_placement_score(
        self,
        pattern: Pattern,
        pattern_poly: Polygon,
        cloth: ClothMaterial,
        placed_patterns: List[PlacedPattern],
    ) -> float:
        """
        Calculate a score for pattern placement quality.
        Higher score = better placement.
        """
        score = 0.0

        # Get pattern bounds
        minx, miny, maxx, maxy = pattern_poly.bounds

        # 1. Edge placement bonus (efficient material usage)
        edge_distances = [
            minx,  # Distance to left edge
            miny,  # Distance to top edge
            cloth.total_width - maxx,  # Distance to right edge
            cloth.total_height - maxy,  # Distance to bottom edge
        ]
        min_edge_dist = min(edge_distances)

        if min_edge_dist < 10:  # Within 10cm of edge
            score += FITTING["REWARDS"]["edge_bonus"] * (1 - min_edge_dist / 10)

        # 2. Compactness bonus (patterns placed close together)
        if placed_patterns:
            min_distance = float("inf")
            for placed in placed_patterns:
                dist = pattern_poly.distance(placed.actual_polygon)
                min_distance = min(min_distance, dist)

            if min_distance < 5:  # Within 5cm of another pattern
                score += FITTING["REWARDS"]["compactness_bonus"] * (
                    1 - min_distance / 5
                )

        # 3. Material utilization bonus
        utilization = pattern.area / cloth.usable_area
        score += FITTING["REWARDS"]["utilization_bonus"] * utilization

        # 4. Avoid creating unusable gaps
        # Check if placement creates small isolated areas
        cloth_poly = Polygon(cloth.contour.squeeze())
        remaining = cloth_poly

        for placed in placed_patterns:
            remaining = remaining.difference(placed.actual_polygon)
        remaining = remaining.difference(pattern_poly)

        # Check for small disconnected regions
        if hasattr(remaining, "geoms"):  # MultiPolygon
            for geom in remaining.geoms:
                if geom.area < FITTING["MIN_GAP_SIZE"] ** 2:
                    score += FITTING["REWARDS"]["gap_penalty"]

        return score

    def create_state_vector(
        self,
        pattern: Pattern,
        cloth: ClothMaterial,
        placed_patterns: List[PlacedPattern],
    ) -> np.ndarray:
        """
        Create state vector for the neural network.
        Represents current fitting situation.
        """
        state = np.zeros(20)  # Fixed size state vector

        # Pattern features
        state[0] = pattern.width / cloth.total_width  # Relative width
        state[1] = pattern.height / cloth.total_height  # Relative height
        state[2] = pattern.area / cloth.usable_area  # Relative area
        state[3] = pattern.width / pattern.height  # Aspect ratio

        # Cloth features
        state[4] = cloth.usable_area / (
            cloth.total_width * cloth.total_height
        )  # Efficiency
        state[5] = len(cloth.defects) / 10.0  # Normalized defect count

        # Placement status
        total_placed_area = sum(p.pattern.area for p in placed_patterns)
        state[6] = total_placed_area / cloth.usable_area  # Utilization so far
        state[7] = len(placed_patterns) / 20.0  # Normalized pattern count

        # Remaining space analysis (simplified)
        if placed_patterns:
            # Approximate remaining space in quadrants
            for i, (qx, qy) in enumerate([(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]):
                # Check if quadrant has patterns
                has_pattern = any(
                    qx * cloth.total_width
                    <= p.position[0]
                    <= (qx + 0.5) * cloth.total_width
                    and qy * cloth.total_height
                    <= p.position[1]
                    <= (qy + 0.5) * cloth.total_height
                    for p in placed_patterns
                )
                state[8 + i] = 0.0 if has_pattern else 1.0
        else:
            state[8:12] = 1.0  # All quadrants available

        # Pattern type encoding (one-hot simplified)
        type_idx = hash(pattern.pattern_type) % 8
        state[12 + type_idx] = 1.0

        return state

    def find_best_placement(
        self,
        pattern: Pattern,
        cloth: ClothMaterial,
        placed_patterns: List[PlacedPattern],
        use_neural_network: bool = True,
    ) -> Optional[PlacedPattern]:
        """
        Find the best placement for a pattern considering all orientations.
        """
        best_placement = None
        best_score = -float("inf")

        # Create state for neural network
        state = self.create_state_vector(pattern, cloth, placed_patterns)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get neural network suggestion
        if use_neural_network:
            self.optimizer.eval()
            with torch.no_grad():
                action, value = self.optimizer(state_tensor)
                action = action.cpu().numpy()[0]

            # Decode action
            suggested_x = action[0] * cloth.total_width
            suggested_y = action[1] * cloth.total_height
            suggested_rotation_idx = int(action[2] * len(self.rotation_angles))
            suggested_flip = action[3] > 0.5

            # Try suggested placement first
            suggestions = [
                (
                    suggested_x,
                    suggested_y,
                    self.rotation_angles[suggested_rotation_idx],
                    suggested_flip,
                ),
            ]
        else:
            suggestions = []

        # Grid search for placement (as backup or primary method)
        search_positions = []

        # Add neural network suggestion
        search_positions.extend(suggestions)

        # Grid-based search
        grid_size = FITTING["GRID_SIZE"]
        x_step = cloth.total_width / grid_size
        y_step = cloth.total_height / grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                x = i * x_step
                y = j * y_step

                # Try all rotations and flips
                for rotation in self.rotation_angles:
                    for flip in [False, True]:
                        search_positions.append((x, y, rotation, flip))

        # Limit search to reasonable number
        if len(search_positions) > FITTING["GRID_SAMPLE_SIZE"]:
            # Random sampling
            indices = np.random.choice(
                len(search_positions), FITTING["GRID_SAMPLE_SIZE"], replace=False
            )
            search_positions = [search_positions[i] for i in indices]

        # Try each position
        attempts = 0
        for x, y, rotation, flip in search_positions:
            if attempts >= FITTING["MAX_ATTEMPTS"]:
                break

            # Create pattern polygon at this position
            pattern_poly = self.create_pattern_polygon(pattern, x, y, rotation, flip)

            # Check validity
            is_valid, coverage = self.check_placement_validity(
                pattern_poly, cloth, placed_patterns
            )

            if is_valid:
                # Calculate placement score
                score = self.calculate_placement_score(
                    pattern, pattern_poly, cloth, placed_patterns
                )

                if score > best_score:
                    best_score = score
                    best_placement = PlacedPattern(
                        pattern=pattern,
                        position=(x, y),
                        rotation=rotation,
                        flipped=flip,
                        placement_score=score,
                        actual_polygon=pattern_poly,
                    )

            attempts += 1

        if best_placement:
            logger.info(
                f"Best placement for {pattern.name}: position=({best_placement.position[0]:.1f}, "
                f"{best_placement.position[1]:.1f}), rotation={best_placement.rotation}°, "
                f"flipped={best_placement.flipped}, score={best_placement.placement_score:.2f}"
            )
        else:
            logger.warning(f"Could not find valid placement for {pattern.name}")

        return best_placement

    def fit_patterns(
        self, patterns: List[Pattern], cloth: ClothMaterial, visualize: bool = True
    ) -> Dict:
        """
        Fit multiple patterns onto cloth material.
        Returns fitting results with metrics.
        """
        logger.info(f"Starting enhanced pattern fitting: {len(patterns)} patterns")
        logger.info(
            f"Cloth: {cloth.total_width:.1f}x{cloth.total_height:.1f} cm, "
            f"usable area: {cloth.usable_area:.1f} cm²"
        )

        # Sort patterns by area (largest first, typically better for packing)
        patterns_sorted = sorted(patterns, key=lambda p: p.area, reverse=True)

        placed_patterns = []
        failed_patterns = []

        # Try to place each pattern
        for i, pattern in enumerate(patterns_sorted):
            logger.info(
                f"\nPlacing pattern {i + 1}/{len(patterns)}: {pattern.name} "
                f"({pattern.pattern_type}, {pattern.width:.1f}x{pattern.height:.1f} cm)"
            )

            placement = self.find_best_placement(pattern, cloth, placed_patterns)

            if placement:
                placed_patterns.append(placement)
                logger.info(f"✓ Successfully placed {pattern.name}")
            else:
                failed_patterns.append(pattern)
                logger.warning(f"✗ Failed to place {pattern.name}")

        # Calculate metrics
        total_pattern_area = sum(p.pattern.area for p in placed_patterns)
        utilization = (total_pattern_area / cloth.usable_area) * 100
        success_rate = (len(placed_patterns) / len(patterns)) * 100

        # Calculate actual material waste
        cloth_poly = Polygon(cloth.contour.squeeze())
        remaining_area = cloth_poly.area

        for placed in placed_patterns:
            remaining_area -= placed.actual_polygon.area

        waste_area = cloth.usable_area - total_pattern_area

        result = {
            "placed_patterns": placed_patterns,
            "failed_patterns": failed_patterns,
            "utilization_percentage": utilization,
            "success_rate": success_rate,
            "total_pattern_area": total_pattern_area,
            "waste_area": waste_area,
            "cloth_dimensions": (cloth.total_width, cloth.total_height),
            "cloth_usable_area": cloth.usable_area,
            "patterns_placed": len(placed_patterns),
            "patterns_total": len(patterns),
        }

        logger.info(f"\nFitting complete:")
        logger.info(f"- Patterns placed: {len(placed_patterns)}/{len(patterns)}")
        logger.info(f"- Material utilization: {utilization:.1f}%")
        logger.info(f"- Waste area: {waste_area:.1f} cm²")

        return result

    def visualize_fitting(
        self,
        patterns: List[Pattern],
        cloth: ClothMaterial,
        result: Dict,
        output_path: str,
    ):
        """
        Create detailed visualization of the fitting result.
        Shows cloth shape, defects, and placed patterns.
        """
        fig, ax = plt.subplots(1, 1, figsize=VISUALIZATION["FIGURE_SIZE"])

        # Draw cloth outline
        cloth_points = cloth.contour.squeeze()
        cloth_patch = MPLPolygon(
            cloth_points,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=2,
            alpha=0.3,
            label="Cloth material",
        )
        ax.add_patch(cloth_patch)

        # Draw defects
        for i, defect in enumerate(cloth.defects):
            if len(defect) > 2:
                defect_points = defect.squeeze()
                defect_patch = MPLPolygon(
                    defect_points,
                    facecolor="red",
                    edgecolor="darkred",
                    alpha=0.8,
                    label="Defect" if i == 0 else "",
                )
                ax.add_patch(defect_patch)

        # Draw placed patterns
        colors = plt.cm.rainbow(np.linspace(0, 1, len(patterns)))

        for i, placed in enumerate(result["placed_patterns"]):
            # Get pattern polygon points
            x, y = placed.actual_polygon.exterior.xy

            # Create patch
            pattern_patch = MPLPolygon(
                list(zip(x, y)),
                facecolor=colors[i],
                edgecolor="black",
                linewidth=VISUALIZATION["LINE_WIDTH"],
                alpha=VISUALIZATION["ALPHA"],
            )
            ax.add_patch(pattern_patch)

            # Add pattern label
            centroid = placed.actual_polygon.centroid
            label_text = f"{placed.pattern.name}\n{placed.pattern.pattern_type}"
            if placed.flipped:
                label_text += " (F)"
            ax.text(
                centroid.x,
                centroid.y,
                label_text,
                ha="center",
                va="center",
                fontsize=VISUALIZATION["FONT_SIZE"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Draw failed patterns (semi-transparent) at the side
        if result["failed_patterns"]:
            y_offset = 10
            for pattern in result["failed_patterns"]:
                # Draw at right side
                x = cloth.total_width + 10

                # Simple rectangle representation
                rect = patches.Rectangle(
                    (x, y_offset),
                    pattern.width,
                    pattern.height,
                    facecolor="gray",
                    edgecolor="red",
                    alpha=0.5,
                    linewidth=2,
                    linestyle="--",
                )
                ax.add_patch(rect)

                # Label
                ax.text(
                    x + pattern.width / 2,
                    y_offset + pattern.height / 2,
                    f"{pattern.name}\n(Failed)",
                    ha="center",
                    va="center",
                    fontsize=VISUALIZATION["FONT_SIZE"] - 2,
                    color="red",
                )

                y_offset += pattern.height + 10

        # Set axis properties
        ax.set_xlim(-10, max(cloth.total_width + 100, cloth.total_width * 1.2))
        ax.set_ylim(-10, max(cloth.total_height + 10, cloth.total_height * 1.1))
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Invert y-axis to match image coordinates

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Width (cm)")
        ax.set_ylabel("Height (cm)")

        # Title with metrics
        title = f"Enhanced Pattern Fitting Result\n"
        title += f"Material: {cloth.cloth_type} | "
        title += f"Utilization: {result['utilization_percentage']:.1f}% | "
        title += f"Patterns: {result['patterns_placed']}/{result['patterns_total']} | "
        title += f"Waste: {result['waste_area']:.1f} cm²"
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add legend
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=VISUALIZATION["DPI"], bbox_inches="tight")
        plt.close()

        logger.info(f"Visualization saved to {output_path}")

        # Also save detailed report
        report_path = output_path.replace(".png", "_report.txt")
        self.save_detailed_report(result, cloth, report_path)

    def save_detailed_report(self, result: Dict, cloth: ClothMaterial, path: str):
        """Save detailed fitting report with all metrics and placement info."""
        with open(path, "w") as f:
            f.write("ENHANCED PATTERN FITTING REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("CLOTH INFORMATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"Type: {cloth.cloth_type}\n")
            f.write(
                f"Dimensions: {cloth.total_width:.1f} x {cloth.total_height:.1f} cm\n"
            )
            f.write(f"Usable Area: {cloth.usable_area:.1f} cm²\n")
            f.write(f"Defects: {len(cloth.defects)}\n")
            f.write(
                f"Material Efficiency: {(cloth.usable_area / (cloth.total_width * cloth.total_height) * 100):.1f}%\n\n"
            )

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
            for i, placed in enumerate(result["placed_patterns"]):
                f.write(f"\nPattern {i + 1}: {placed.pattern.name}\n")
                f.write(f"  Type: {placed.pattern.pattern_type}\n")
                f.write(
                    f"  Size: {placed.pattern.width:.1f} x {placed.pattern.height:.1f} cm\n"
                )
                f.write(f"  Area: {placed.pattern.area:.1f} cm²\n")
                f.write(
                    f"  Position: ({placed.position[0]:.1f}, {placed.position[1]:.1f}) cm\n"
                )
                f.write(f"  Rotation: {placed.rotation}°\n")
                f.write(f"  Flipped: {'Yes' if placed.flipped else 'No'}\n")
                f.write(f"  Placement Score: {placed.placement_score:.2f}\n")

            if result["failed_patterns"]:
                f.write("\nFAILED PATTERNS\n")
                f.write("-" * 60 + "\n")
                for pattern in result["failed_patterns"]:
                    f.write(f"- {pattern.name} ({pattern.pattern_type}): ")
                    f.write(f"{pattern.width:.1f} x {pattern.height:.1f} cm\n")

        logger.info(f"Detailed report saved to {path}")

    def train_on_placement(self, state: np.ndarray, action: Tuple, reward: float):
        """Update the neural network based on placement result."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Convert action to tensor
        action_tensor = (
            torch.FloatTensor(
                [
                    action[0] / 200.0,  # Normalize x
                    action[1] / 300.0,  # Normalize y
                    self.rotation_angles.index(action[2]) / len(self.rotation_angles),
                    float(action[3]),
                ]
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # Forward pass
        predicted_action, value = self.optimizer(state_tensor)

        # Calculate loss
        action_loss = torch.nn.functional.mse_loss(predicted_action, action_tensor)
        value_loss = torch.nn.functional.mse_loss(
            value, torch.tensor([[reward]], device=self.device)
        )

        total_loss = action_loss + value_loss

        # Backward pass
        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()

    def save_model(self):
        """Save the placement optimizer model."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        checkpoint = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "rotation_angles": self.rotation_angles,
        }

        with open(self.model_path, "wb") as f:
            pickle.dump(checkpoint, f)

        logger.info(f"Placement optimizer saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load the placement optimizer model."""
        if os.path.exists(self.model_path):
            logger.info(f"Loading placement optimizer from {self.model_path}")

            with open(self.model_path, "rb") as f:
                checkpoint = pickle.load(f)

            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.optim.load_state_dict(checkpoint["optim_state_dict"])

            if "rotation_angles" in checkpoint:
                self.rotation_angles = checkpoint["rotation_angles"]

            logger.info("Placement optimizer loaded successfully")
            return True
        return False
