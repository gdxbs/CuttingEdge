"""
Simple Pattern Fitting Module
Uses basic reinforcement learning to fit patterns onto cloth.
Designed to be easy to understand for beginners.
"""

import os
import cv2
import numpy as np
import torch
import pickle
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .simple_pattern_recognition import Pattern

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Placement:
    """Represents where a pattern is placed on cloth"""

    pattern_id: int
    x: float  # Top-left x coordinate
    y: float  # Top-left y coordinate
    rotation: float  # Rotation angle in degrees


class SimpleFittingAgent:
    """
    A simple agent that learns to place patterns on cloth.
    Uses a basic neural network instead of complex RL algorithms.
    """

    def __init__(self):
        # Simple network that takes state and outputs action
        self.net = torch.nn.Sequential(
            torch.nn.Linear(10, 64),  # Input: simplified state
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3),  # Output: x, y, rotation
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def get_action(self, state: np.ndarray) -> Tuple[float, float, float]:
        """
        Given a state, return where to place the pattern.
        Returns: (x, y, rotation)
        """
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            output = self.net(state_tensor)

        # Convert network output to actual placement
        x = torch.sigmoid(output[0]).item()  # 0-1 range
        y = torch.sigmoid(output[1]).item()  # 0-1 range
        rotation = torch.tanh(output[2]).item() * 45  # -45 to 45 degrees

        return x, y, rotation

    def update(self, state: np.ndarray, action: Tuple, reward: float):
        """
        Update the network based on the reward received.
        This is a simplified learning update.
        """
        state_tensor = torch.FloatTensor(state)
        output = self.net(state_tensor)

        # Simple loss based on reward (higher reward = lower loss)
        loss = -reward * output.mean()  # Simplified loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PatternFitter:
    """
    Main class for fitting patterns onto cloth.
    Uses simple algorithms that are easy to understand.
    """

    # Magic numbers for fitting
    GRID_SIZE = 10  # Divide cloth into 10x10 grid for placement
    MAX_ATTEMPTS = 100  # Max attempts to place a pattern
    OVERLAP_PENALTY = -10  # Penalty for overlapping patterns
    EDGE_BONUS = 2  # Bonus for placing near edges (efficient use)
    UTILIZATION_BONUS = 5  # Bonus for good space utilization

    def __init__(self, model_path: str = "models/fitting_model.pkl"):
        self.model_path = model_path
        self.agent = SimpleFittingAgent()
        self.placements = []

        logger.info("Pattern Fitter initialized")

    def create_state(
        self, cloth_info: Dict, pattern: Pattern, current_placements: List[Placement]
    ) -> np.ndarray:
        """
        Create a simplified state representation for the agent.
        This makes it easier for the network to learn.
        """
        cloth_w, cloth_h = cloth_info["usable_dimensions"]

        # State features (kept simple):
        state = np.zeros(10)

        # 1. Pattern size relative to cloth
        state[0] = pattern.width / cloth_w
        state[1] = pattern.height / cloth_h

        # 2. How much cloth is already used
        used_area = sum([self.calculate_pattern_area(p) for p in current_placements])
        state[2] = used_area / (cloth_w * cloth_h)

        # 3. Number of patterns already placed
        state[3] = len(current_placements) / 10.0  # Normalize

        # 4. Aspect ratios
        state[4] = pattern.width / pattern.height
        state[5] = cloth_w / cloth_h

        # 5. Available space indicators (simplified)
        # Check corners and center
        state[6] = (
            1.0
            if self.is_space_available(0, 0, pattern, current_placements, cloth_info)
            else 0.0
        )
        state[7] = (
            1.0
            if self.is_space_available(
                cloth_w - pattern.width, 0, pattern, current_placements, cloth_info
            )
            else 0.0
        )
        state[8] = (
            1.0
            if self.is_space_available(
                0, cloth_h - pattern.height, pattern, current_placements, cloth_info
            )
            else 0.0
        )
        state[9] = (
            1.0
            if self.is_space_available(
                cloth_w / 2, cloth_h / 2, pattern, current_placements, cloth_info
            )
            else 0.0
        )

        return state

    def calculate_pattern_area(self, placement: Placement) -> float:
        """Calculate area of a placed pattern (simplified)"""
        # For now, return a fixed area
        # In real implementation, would calculate based on pattern contour
        return 100.0  # Simplified

    def is_space_available(
        self,
        x: float,
        y: float,
        pattern: Pattern,
        placements: List[Placement],
        cloth_info: Dict,
    ) -> bool:
        """
        Check if space is available at given position.
        Simplified collision detection.
        """
        cloth_w, cloth_h = cloth_info["usable_dimensions"]

        # Check bounds
        if (
            x < 0
            or y < 0
            or x + pattern.width > cloth_w
            or y + pattern.height > cloth_h
        ):
            return False

        # Check overlap with other patterns (simplified)
        for p in placements:
            # Simple rectangle overlap check (ignoring rotation for simplicity)
            if (
                x < p.x + pattern.width
                and x + pattern.width > p.x
                and y < p.y + pattern.height
                and y + pattern.height > p.y
            ):
                return False

        return True

    def calculate_reward(
        self,
        placement: Placement,
        pattern: Pattern,
        cloth_info: Dict,
        all_placements: List[Placement],
    ) -> float:
        """
        Calculate reward for a placement.
        Higher reward = better placement.
        """
        reward = 0.0
        cloth_w, cloth_h = cloth_info["usable_dimensions"]

        # Check if placement is valid
        if not self.is_space_available(
            placement.x, placement.y, pattern, all_placements[:-1], cloth_info
        ):  # Exclude current
            return self.OVERLAP_PENALTY

        # Reward for edge placement (efficient packing)
        edge_distance = min(
            placement.x,
            placement.y,
            cloth_w - (placement.x + pattern.width),
            cloth_h - (placement.y + pattern.height),
        )
        if edge_distance < 10:  # Close to edge
            reward += self.EDGE_BONUS

        # Reward for space utilization
        utilization = (pattern.width * pattern.height) / (cloth_w * cloth_h)
        reward += utilization * self.UTILIZATION_BONUS

        # Penalty for wasted space (simplified)
        # Check if the placement creates unusable small gaps
        min_gap = 20  # Minimum useful gap size
        if placement.x > 0 and placement.x < min_gap:
            reward -= 1
        if placement.y > 0 and placement.y < min_gap:
            reward -= 1

        return reward

    def fit_patterns(self, patterns: List[Pattern], cloth_info: Dict) -> Dict:
        """
        Main method to fit patterns onto cloth.
        Returns fitting results with metrics.
        """
        logger.info(f"Starting pattern fitting: {len(patterns)} patterns")

        cloth_w, cloth_h = cloth_info["usable_dimensions"]
        placements = []
        total_pattern_area = 0

        # Try to place each pattern
        for i, pattern in enumerate(patterns):
            logger.info(f"Placing pattern {i + 1}/{len(patterns)}: {pattern.name}")

            best_placement = None
            best_reward = -float("inf")

            # Try multiple placements
            for attempt in range(self.MAX_ATTEMPTS):
                # Get state
                state = self.create_state(cloth_info, pattern, placements)

                # Get action from agent
                x_norm, y_norm, rotation = self.agent.get_action(state)

                # Convert normalized coordinates to actual
                x = x_norm * (cloth_w - pattern.width)
                y = y_norm * (cloth_h - pattern.height)

                # Create placement
                placement = Placement(pattern_id=i, x=x, y=y, rotation=rotation)

                # Calculate reward
                temp_placements = placements + [placement]
                reward = self.calculate_reward(
                    placement, pattern, cloth_info, temp_placements
                )

                # Update agent
                self.agent.update(state, (x_norm, y_norm, rotation), reward)

                # Track best placement
                if reward > best_reward:
                    best_reward = reward
                    best_placement = placement

                # If we found a good placement, stop trying
                if reward > 0:
                    break

            # Place the pattern if we found a valid spot
            if best_placement and best_reward > self.OVERLAP_PENALTY:
                placements.append(best_placement)
                total_pattern_area += pattern.width * pattern.height
                logger.info(
                    f"  Placed at ({best_placement.x:.1f}, {best_placement.y:.1f}) "
                    f"with rotation {best_placement.rotation:.1f}°"
                )
            else:
                logger.warning(f"  Could not place pattern {pattern.name}")

        # Calculate metrics
        utilization = (total_pattern_area / cloth_info["usable_area"]) * 100
        patterns_placed = len(placements)
        patterns_total = len(patterns)

        result = {
            "placements": placements,
            "utilization_percentage": utilization,
            "patterns_placed": patterns_placed,
            "patterns_total": patterns_total,
            "success_rate": (patterns_placed / patterns_total) * 100,
            "cloth_dimensions": cloth_info["usable_dimensions"],
            "total_pattern_area": total_pattern_area,
            "wasted_area": cloth_info["usable_area"] - total_pattern_area,
        }

        logger.info(
            f"Fitting complete: {patterns_placed}/{patterns_total} patterns placed"
        )
        logger.info(f"Material utilization: {utilization:.1f}%")

        return result

    def visualize_result(
        self, patterns: List[Pattern], cloth_info: Dict, result: Dict, output_path: str
    ):
        """
        Create a visual representation of the fitting result.
        """
        cloth_w, cloth_h = cloth_info["usable_dimensions"]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Draw cloth background
        cloth_rect = patches.Rectangle(
            (0, 0),
            cloth_w,
            cloth_h,
            linewidth=2,
            edgecolor="black",
            facecolor="lightgray",
            alpha=0.3,
        )
        ax.add_patch(cloth_rect)

        # Draw placed patterns
        colors = plt.cm.rainbow(np.linspace(0, 1, len(patterns)))

        for placement in result["placements"]:
            pattern = patterns[placement.pattern_id]

            # Create rectangle for pattern
            rect = patches.Rectangle(
                (placement.x, placement.y),
                pattern.width,
                pattern.height,
                linewidth=2,
                edgecolor="black",
                facecolor=colors[placement.pattern_id],
                alpha=0.7,
                angle=placement.rotation,
            )
            ax.add_patch(rect)

            # Add pattern name
            ax.text(
                placement.x + pattern.width / 2,
                placement.y + pattern.height / 2,
                pattern.name,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        # Set axis properties
        ax.set_xlim(0, cloth_w)
        ax.set_ylim(0, cloth_h)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Invert y-axis to match image coordinates

        # Add title and labels
        ax.set_title(
            f"Pattern Fitting Result\n"
            f"Utilization: {result['utilization_percentage']:.1f}% | "
            f"Patterns: {result['patterns_placed']}/{result['patterns_total']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Width (cm)")
        ax.set_ylabel("Height (cm)")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Visualization saved to {output_path}")

        # Also save a detailed report
        report_path = output_path.replace(".png", "_report.txt")
        self.save_report(result, report_path)

        plt.close()

    def save_report(self, result: Dict, path: str):
        """Save detailed fitting report"""
        with open(path, "w") as f:
            f.write("PATTERN FITTING REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total Patterns: {result['patterns_total']}\n")
            f.write(f"Patterns Placed: {result['patterns_placed']}\n")
            f.write(f"Success Rate: {result['success_rate']:.1f}%\n")
            f.write(f"Material Utilization: {result['utilization_percentage']:.1f}%\n")
            f.write(
                f"Cloth Dimensions: {result['cloth_dimensions'][0]:.1f} x {result['cloth_dimensions'][1]:.1f} cm\n"
            )
            f.write(f"Total Pattern Area: {result['total_pattern_area']:.1f} cm²\n")
            f.write(f"Wasted Area: {result['wasted_area']:.1f} cm²\n\n")

            f.write("PLACEMENT DETAILS\n")
            f.write("-" * 50 + "\n")
            for i, p in enumerate(result["placements"]):
                f.write(
                    f"Pattern {i + 1}: Position ({p.x:.1f}, {p.y:.1f}), Rotation {p.rotation:.1f}°\n"
                )

    def save_model(self):
        """Save the fitting model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "agent_state": self.agent.net.state_dict(),
                    "optimizer_state": self.agent.optimizer.state_dict(),
                },
                f,
            )
        logger.info(f"Fitting model saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load existing model if available"""
        if os.path.exists(self.model_path):
            logger.info(f"Loading fitting model from {self.model_path}")
            with open(self.model_path, "rb") as f:
                checkpoint = pickle.load(f)
                self.agent.net.load_state_dict(checkpoint["agent_state"])
                self.agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
            logger.info("Fitting model loaded successfully!")
            return True
        return False
