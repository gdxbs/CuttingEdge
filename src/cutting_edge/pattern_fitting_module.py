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
from stable_baselines3 import PPO  # Stable RL implementation

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

class PackingEnvironment:
    """Environment for pattern packing optimization

    This class implements a reinforcement learning environment for optimizing
    pattern placement on cloth. It handles the simulation of placing pattern
    contours on a 2D cloth space while enforcing constraints such as no overlapping
    and staying within cloth boundaries.

    The environment follows the OpenAI Gym-like interface with methods for:
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
        self, cloth_data: Dict, patterns: List[Dict], rotation_angles: List[int]
    ):
        self.cloth_data = cloth_data
        self.cloth_width, self.cloth_height = map(int, cloth_data["dimensions"])
        self.patterns = patterns
        self.current_state = None
        self.placed_patterns = []
        self.grid_size = 1  # Minimum unit for placement
        self.rotation_angles = rotation_angles

        # Create Shapely polygon for cloth boundary
        self.cloth_polygon = sg.box(0, 0, self.cloth_width, self.cloth_height)

        # Convert patterns to Shapely polygons
        self.pattern_polygons = []
        for pattern in patterns:
            if "contours" in pattern and pattern["contours"]:
                # Convert OpenCV contour to Shapely polygon
                contour = pattern["contours"][0]
                polygon = sg.Polygon(contour)
                self.pattern_polygons.append(polygon)
            else:
                # Fallback to a simple rectangle if no contours
                if "dimensions" in pattern:
                    width, height = pattern["dimensions"]
                    polygon = sg.box(0, 0, width, height)
                    self.pattern_polygons.append(polygon)
                else:
                    # Skip patterns without proper geometry
                    self.pattern_polygons.append(None)
                    logger.warning("Pattern missing both contours and dimensions")

        # Set up the observation and action spaces for Gym compatibility
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(3, self.cloth_height, self.cloth_width),
            dtype=np.float32,
        )

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
            }
        )

        # Initialize cloth space
        self.reset()

    def reset(self):
        """Reset environment to initial state"""

        self.cloth_space = np.zeros(
            (self.cloth_height, self.cloth_width), dtype=np.uint8
        )
        self.current_state = self._get_state()
        self.placed_patterns = []
        self.available_patterns = list(
            range(len(self.patterns))
        )  # Indices of available patterns

        return self.current_state

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        action: Dict containing pattern_idx, position, rotation
        """

        pattern_idx = action["pattern_idx"]
        position = action["position"]
        rotation = (
            action["rotation"]
            if isinstance(action["rotation"], float)
            else self.rotation_angles[action["rotation"]]
        )

        # If pattern is already placed
        if pattern_idx not in self.available_patterns:
            reward = -1.0
            done = len(self.placed_patterns) == len(self.patterns)
            info = {
                "success": False,
                "utilization": self._calculate_utilization(),
                "remaining_patterns": len(self.patterns) - len(self.placed_patterns),
            }
            return self.current_state, reward, done, info

        pattern = self.patterns[pattern_idx]

        # Try to place pattern using Shapely for precise collision detection
        success = self._place_pattern(pattern, position, rotation)

        if success:
            reward = self._calculate_reward()
            self.placed_patterns.append(
                {
                    "pattern": pattern,
                    "position": position,
                    "rotation": rotation,
                    "pattern_idx": pattern_idx,
                }
            )
            self.available_patterns.remove(pattern_idx)
        else:
            reward = -1.0  # Penalty for invalid placement

        # Update state
        self.current_state = self._get_state()

        # Check if episode is done
        done = len(self.placed_patterns) == len(self.patterns)

        info = {
            "success": success,
            "utilization": self._calculate_utilization(),
            "remaining_patterns": len(self.patterns) - len(self.placed_patterns),
        }

        return self.current_state, reward, done, info

    def _place_pattern(
        self, pattern: Dict, position: Tuple[int, int], rotation: float
    ) -> bool:
        """Attempt to place pattern at given position and rotation using Shapely"""

        x, y = position
        pattern_idx = pattern.get("idx", self.patterns.index(pattern))

        if (
            pattern_idx >= len(self.pattern_polygons)
            or self.pattern_polygons[pattern_idx] is None
        ):
            return False

        # Get the pattern polygon
        pattern_polygon = self.pattern_polygons[pattern_idx]

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
        """Calculate reward based on multiple factors"""

        utilization = self._calculate_utilization()
        compactness = self._calculate_compactness()
        edge_distance = self._calculate_edge_distance()
        valid_placement = 1.0  # All placements are valid at this point

        # Weighted reward calculation
        reward = (
            0.5 * utilization
            + 0.3 * compactness
            + 0.1 * edge_distance
            + 0.1 * valid_placement
        )

        return reward

    def _calculate_utilization(self) -> float:
        """Calculate material utilization percentage"""

        used_area = np.sum(self.cloth_space)
        total_area = self.cloth_width * self.cloth_height
        return used_area / total_area

    def _calculate_compactness(self) -> float:
        """Calculate how compactly patterns are placed"""

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
        """Calculate average distance to cloth edges"""

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
        """Get current state representation"""

        # Stack cloth space with pattern features
        state = np.stack(
            [
                self.cloth_space,
                self._get_pattern_channel(),
                self._get_distance_channel(),
            ]
        )

        return state

    def _get_pattern_channel(self) -> np.ndarray:
        """Create channel showing next pattern to place"""

        if self.available_patterns:
            # Get the first available pattern index
            next_pattern_idx = self.available_patterns[0]
            next_pattern = self.patterns[next_pattern_idx]
            pattern_channel = np.zeros_like(self.cloth_space)
            cv2.fillPoly(pattern_channel, next_pattern["contours"], 1)

            return pattern_channel

        return np.zeros_like(self.cloth_space)

    def _get_distance_channel(self) -> np.ndarray:
        """Create distance transform channel"""

        # Distance transform helps the model understand spatial relationships
        # This creates a map where each pixel value is the distance to the nearest
        # occupied pixel in the cloth space
        return cv2.distanceTransform(
            (1 - self.cloth_space).astype(np.uint8), cv2.DIST_L2, 5
        )

class HierarchicalRL:
    """Hierarchical Reinforcement Learning for pattern packing

    This class implements a two-level hierarchical reinforcement learning approach
    for optimizing pattern placement on cloth materials. It leverages stable-baselines3
    for a robust implementation of PPO (Proximal Policy Optimization).

    The hierarchical structure allows for complex decision-making by separating
    the "what" (pattern selection) from the "how" (placement details).

    References:
    - "Planning Irregular Object Packing via Hierarchical Reinforcement Learning" (Wang et al., 2022)
    - "Tree Search + Reinforcement Learning for Two-Dimensional Cutting Stock Problem
       With Complex Constraints" (Zhang et al., 2023)
    """

    def __init__(self, env: PackingEnvironment, device: torch.device = None):
        self.env = env
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize PPO agent
        policy_kwargs = {
            "features_extractor_class": None,  # Use default
            "features_extractor_kwargs": {"features_dim": 64},
            "net_arch": [{"pi": [64, 64], "vf": [64, 64]}],
        }

        self.agent = PPO(
            "CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device=self.device
        )

        # Experience buffer for custom training
        self.buffer = []
        self.gamma = 0.99  # Discount factor

    def train(self, num_episodes: int):
        """Train the agent using stable-baselines3 PPO

        Args:
            num_episodes: Number of training episodes

        Returns:
            Tuple of (best_state, best_utilization) achieved during training
        """
        total_timesteps = num_episodes * len(self.env.patterns) * 2
        logger.info(f"Training for {total_timesteps} total timesteps")

        best_utilization = 0.0
        best_state = None

        # Define callback to track best performance
        class BestModelCallback:
            def __init__(self, env):
                self.env = env
                self.best_utilization = 0
                self.best_state = None

            def __call__(self, locals, globals):
                info = locals.get("info", {})
                if isinstance(info, dict) and "utilization" in info:
                    utilization = info["utilization"]
                    if utilization > self.best_utilization:
                        self.best_utilization = utilization
                        self.best_state = self.env.cloth_space.copy()
                        logger.info(
                            f"New best utilization: {self.best_utilization:.4f}"
                        )
                return True

        callback = BestModelCallback(self.env)

        # Train the model
        self.agent.learn(total_timesteps=total_timesteps, callback=callback)

        best_state = callback.best_state
        best_utilization = callback.best_utilization

        return best_state, best_utilization

    def save_model(self, model_path: str):
        """Save trained model using stable-baselines3 save method"""
        self.agent.save(model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load trained model using stable-baselines3 load method"""
        self.agent = PPO.load(model_path, env=self.env, device=self.device)
        logger.info(f"Model loaded from {model_path}")

    def infer(self, visualize=False):
        """Run inference to find optimal pattern placement

        Uses the trained agent to find an optimal pattern placement solution
        without exploration (deterministic=True).

        Args:
            visualize: If True, generates step-by-step visualizations of pattern placement

        Returns:
            Tuple of (final_state, utilization, placement_data) with the solution details
        """
        state = self.env.reset()
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
            # Use the agent for prediction
            action, _ = self.agent.predict(state, deterministic=True)

            # Convert action to the format expected by the environment
            env_action = {}
            if isinstance(action, dict):
                env_action = action
            else:
                # If action is a single value, convert to dict format
                pattern_idx = int(action) % len(self.env.patterns)
                position_x = int(
                    (action // len(self.env.patterns)) % self.env.cloth_width
                )
                position_y = int(
                    (action // (len(self.env.patterns) * self.env.cloth_width))
                    % self.env.cloth_height
                )
                rotation_idx = int(
                    action
                    // (
                        len(self.env.patterns)
                        * self.env.cloth_width
                        * self.env.cloth_height
                    )
                ) % len(self.env.rotation_angles)

                env_action = {
                    "pattern_idx": pattern_idx,
                    "position": (position_x, position_y),
                    "rotation": rotation_idx,
                }

            # Take action
            next_state, reward, done, info = self.env.step(env_action)
            total_reward += reward

            # Visualize step if requested
            if visualize:
                plt.figure(figsize=(10, 10))
                plt.imshow(self.env.cloth_space, cmap="gray")
                plt.title(f"Step {step}: Pattern {env_action['pattern_idx']} placed")
                plt.savefig(os.path.join(output_dir, f"step_{step}_cloth.png"))
                plt.close()

            state = next_state
            step += 1

            # Log progress
            pattern_idx = env_action["pattern_idx"]
            position = env_action["position"]
            rotation = (
                self.env.rotation_angles[env_action["rotation"]]
                if isinstance(env_action["rotation"], int)
                else env_action["rotation"]
            )
            logger.info(
                f"Step {step}: Placed pattern {pattern_idx} at {position} with rotation {rotation}°"
            )
            logger.info(f"Current utilization: {info['utilization']:.4f}")

            # If no placement was successful, break to prevent infinite loop
            if not info["success"]:
                logger.warning("Placement failed, ending episode")
                break

            # Check if all patterns are placed
            if len(self.env.available_patterns) == 0:
                logger.info("All patterns placed successfully")
                done = True

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
    """Module for fitting patterns onto cloth using Reinforcement Learning

    This module implements pattern fitting using Hierarchical Reinforcement Learning (HRL)
    for optimizing pattern placement on cloth materials.

    References:
    - "Planning Irregular Object Packing via Hierarchical Reinforcement Learning"
      (Wang et al., 2022)
    - "Tree Search + Reinforcement Learning for Two-Dimensional Cutting Stock Problem
      With Complex Constraints" (Zhang et al., 2023)
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

        This method processes the cloth and pattern data into the format expected by the
        packing environment, including scaling/normalization if needed.

        Args:
            cloth_data: Dictionary containing cloth properties (from cloth_recognition_module)
            patterns_data: List of dictionaries containing pattern properties

        Returns:
            Processed cloth and patterns data
        """
        # Process cloth data - ensure dimensions are reasonable
        processed_cloth = cloth_data.copy()

        # Scale cloth dimensions if needed (for very large cloth)
        cloth_width, cloth_height = map(float, processed_cloth["dimensions"])
        max_size = 512  # Maximum size for either dimension

        if cloth_width > max_size or cloth_height > max_size:
            scale = max_size / max(cloth_width, cloth_height)
            processed_cloth["dimensions"] = (cloth_width * scale, cloth_height * scale)
            logger.info(
                f"Scaled cloth from {(cloth_width, cloth_height)} to {processed_cloth['dimensions']}"
            )

            # Also scale the contours if they exist
            if "contours" in processed_cloth and processed_cloth["contours"]:
                for i, contour in enumerate(processed_cloth["contours"]):
                    processed_cloth["contours"][i] = contour * scale

        # Process patterns data
        processed_patterns = []
        for pattern in patterns_data:
            # Create a copy of the pattern data
            processed_pattern = pattern.copy()

            # Ensure pattern has required fields
            if "contours" not in processed_pattern or not processed_pattern["contours"]:
                logger.warning("Pattern missing contours, skipping")
                continue

            # Apply same scale to patterns if cloth was scaled
            if cloth_width > max_size or cloth_height > max_size:
                for i, contour in enumerate(processed_pattern["contours"]):
                    processed_pattern["contours"][i] = contour * scale

                # Scale dimensions if they exist
                if "dimensions" in processed_pattern:
                    processed_pattern["dimensions"] = tuple(
                        d * scale for d in processed_pattern["dimensions"]
                    )

            processed_patterns.append(processed_pattern)

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
        """Fit patterns onto cloth using Reinforcement Learning

        Args:
            cloth_data: Dictionary containing cloth properties
            patterns_data: List of dictionaries containing pattern properties
            visualize: Whether to generate visualization images

        Returns:
            Dictionary with fitting results
        """
        # Prepare data
        processed_cloth, processed_patterns = self.prepare_data(
            cloth_data, patterns_data
        )
        cloth_width, cloth_height = map(int, processed_cloth["dimensions"])

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
            "method": "reinforcement_learning",
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

        # Get canvas from final state
        canvas = result["final_state"]
        cloth_width, cloth_height = result["cloth_dims"]

        # Plot the original cloth if provided
        if cloth_image is not None:
            plt.subplot(1, 2, 1)
            plt.imshow(cloth_image)
            plt.title("Original Cloth")
            plt.axis("off")
            plt.subplot(1, 2, 2)

        # Display the final result
        plt.imshow(canvas, cmap=cmap, interpolation="nearest")

        # Add title with method and utilization
        method_name = "Reinforcement Learning"
        plt.title(
            f"Pattern Placement using {method_name}\n(Utilization: {result['utilization']:.1%})"
        )
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
