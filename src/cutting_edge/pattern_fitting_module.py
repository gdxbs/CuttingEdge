import logging
import os
from typing import Dict, List

import cv2
import gymnasium as gym
import numpy as np
import shapely.affinity as sa
import shapely.geometry as sg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Pattern Fitting Module

This module implements pattern fitting onto cloth materials using reinforcement learning.
The system optimizes material utilization by determining:
1. Which pattern to place next
2. Where and at what rotation to place it
"""

class PackingEnvironment(gym.Env):
    """Environment for pattern packing optimization

    A reinforcement learning environment that simulates pattern placement on cloth.
    Handles all the physics and constraints of proper pattern placement.
    """

    def __init__(self, cloth_data: Dict, patterns: List[Dict], rotation_angles: List[int]):
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
        self.placed_patterns = []
        self.available_patterns = list(range(len(patterns)))

        # Create cloth boundary as Shapely polygon
        self.cloth_polygon = sg.box(0, 0, self.cloth_width, self.cloth_height)

        # Process patterns into Shapely polygons for collision detection
        self._process_patterns()

        # Define observation and action spaces
        self._setup_spaces()

        # Initialize the environment
        self.reset()

    def _process_patterns(self):
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
                            contour_points = np.vstack([contour_points, contour_points[0]])

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

    def _create_polygon_from_dimensions(self, pattern: Dict, pattern_idx: int):
        """Create a rectangular polygon from dimensions"""
        if "dimensions" in pattern:
            width, height = pattern["dimensions"]
            if hasattr(width, "item"):  # Handle tensor types
                width, height = width.item(), height.item()
            polygon = sg.box(0, 0, float(width), float(height))
            self.pattern_polygons.append(polygon)
            self.pattern_idx_map.append(pattern_idx)

    def _setup_spaces(self):
        """Define observation and action spaces for RL"""
        # State space: cloth occupancy map + pattern maps
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(3 + len(self.patterns), self.cloth_height, self.cloth_width),
            dtype=np.float32,
        )

        # Action space: pattern_idx, position, rotation
        self.action_space = gym.spaces.Dict({
            "pattern_idx": gym.spaces.Discrete(len(self.patterns)),
            "position": gym.spaces.Box(
                low=np.array([0, 0]),
                high=np.array([self.cloth_width - 1, self.cloth_height - 1]),
                dtype=np.int32,
            ),
            "rotation": gym.spaces.Discrete(len(self.rotation_angles)),
        })

    def reset(self, seed=None):
        """Reset environment to initial state

        Args:
            seed: Random seed for reproducibility

        Returns:
            Current state observation and empty info dict
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Initialize empty cloth space
        self.cloth_space = np.zeros((self.cloth_height, self.cloth_width), dtype=np.uint8)

        # Reset pattern tracking
        self.placed_patterns = []
        self.available_patterns = list(range(len(self.patterns)))

        # Get initial state
        current_state = self._get_state()

        return current_state, {}

    def step(self, action):
        """Execute one step (place one pattern)

        Args:
            action: Dict with pattern_idx, position, and rotation

        Returns:
            state: Updated environment state
            reward: Reward for the action
            done: Whether episode is complete
            truncated: Whether episode was truncated
            info: Additional info dict
        """
        pattern_idx = action["pattern_idx"]
        position = action["position"]
        rotation = self.rotation_angles[action["rotation"]]

        # Handle already placed patterns
        if pattern_idx not in self.available_patterns:
            reward = -1.0
            done = len(self.placed_patterns) == len(self.patterns)
            info = {
                "success": False,
                "utilization": self._calculate_utilization(),
                "remaining_patterns": len(self.patterns) - len(self.placed_patterns),
            }
            return self._get_state(), reward, done, False, info

        # Get the original pattern
        original_pattern_idx = self.pattern_idx_map[pattern_idx] if pattern_idx < len(self.pattern_idx_map) else pattern_idx
        pattern = self.patterns[original_pattern_idx]

        # Try to place the pattern
        success = self._place_pattern(pattern, position, rotation, pattern_idx)

        # Calculate reward and update state
        if success:
            reward = self._calculate_reward()
            self._track_placed_pattern(pattern, position, rotation, pattern_idx)
        else:
            reward = -1.0

        # Get updated state
        current_state = self._get_state()

        # Check if episode is done
        done = len(self.placed_patterns) == len(self.patterns)

        # Add max steps to prevent very long episodes
        max_steps = len(self.patterns) * 3
        truncated = len(self.placed_patterns) > max_steps

        # Create info dictionary
        info = {
            "success": success,
            "utilization": self._calculate_utilization(),
            "remaining_patterns": len(self.patterns) - len(self.placed_patterns),
        }

        return current_state, reward, done, truncated, info

    def _track_placed_pattern(self, pattern, position, rotation, pattern_idx):
        """Track a successfully placed pattern"""
        self.placed_patterns.append({
            "pattern": pattern,
            "position": position,
            "rotation": rotation,
            "pattern_idx": pattern_idx,
        })
        self.available_patterns.remove(pattern_idx)

    def _place_pattern(self, pattern, position, rotation, pattern_polygon_idx=None):
        """Place a pattern at the given position and rotation

        Args:
            pattern: Pattern data dictionary
            position: (x, y) position tuple
            rotation: Rotation angle in degrees
            pattern_polygon_idx: Index of the pattern polygon to use

        Returns:
            bool: Whether the placement was successful
        """
        x, y = position

        # Get the pattern polygon
        polygon_idx = self._get_polygon_index(pattern, pattern_polygon_idx)

        # Return failure if no polygon found
        if polygon_idx == -1 or polygon_idx >= len(self.pattern_polygons):
            return False

        # Get the pattern polygon and transform it
        pattern_polygon = self.pattern_polygons[polygon_idx]
        rotated_polygon = sa.rotate(pattern_polygon, rotation, origin=(0, 0))
        placed_polygon = sa.translate(rotated_polygon, x, y)

        # Check if pattern is within cloth bounds
        if not self.cloth_polygon.contains(placed_polygon):
            return False

        # Check overlap with already placed patterns
        if self._check_overlap(placed_polygon):
            return False

        # Update the cloth space for visualization
        self._update_cloth_space(placed_polygon)

        return True

    def _get_polygon_index(self, pattern, pattern_polygon_idx=None):
        """Get the index of the polygon for a pattern"""
        if pattern_polygon_idx is not None and pattern_polygon_idx < len(self.pattern_polygons):
            return pattern_polygon_idx

        # Try to find matching polygon
        polygon_idx = -1
        for i, p in enumerate(self.patterns):
            if p is pattern:
                for j, mapped_idx in enumerate(self.pattern_idx_map):
                    if mapped_idx == i:
                        polygon_idx = j
                        break
                if polygon_idx >= 0:
                    break

        return polygon_idx

    def _check_overlap(self, new_polygon):
        """Check if a new polygon overlaps with existing patterns"""
        for placed in self.placed_patterns:
            placed_pattern_idx = placed["pattern_idx"]
            placed_pos = placed["position"]
            placed_rot = placed["rotation"]

            if placed_pattern_idx < len(self.pattern_polygons):
                existing_poly = self.pattern_polygons[placed_pattern_idx]
                existing_poly = sa.rotate(existing_poly, placed_rot, origin=(0, 0))
                existing_poly = sa.translate(existing_poly, placed_pos[0], placed_pos[1])

                # Check for intersection
                if new_polygon.intersects(existing_poly):
                    return True

        return False

    def _update_cloth_space(self, placed_polygon):
        """Update the cloth space with the placed pattern"""
        pattern_mask = np.zeros_like(self.cloth_space)

        # Convert Shapely polygon to OpenCV contour
        polygon_points = np.array(placed_polygon.exterior.coords).astype(int)[:-1]
        cv2.fillPoly(pattern_mask, [polygon_points], 1)

        # Place pattern on cloth space
        self.cloth_space = np.logical_or(self.cloth_space, pattern_mask).astype(np.uint8)

    def _calculate_reward(self):
        """Calculate reward based on pattern placement"""
        utilization = self._calculate_utilization()
        compactness = self._calculate_compactness()
        valid_placement = 1.0

        # Weighted reward
        reward = 0.7 * utilization + 0.2 * compactness + 0.1 * valid_placement

        return reward

    def _calculate_utilization(self):
        """Calculate material utilization percentage"""
        used_area = np.sum(self.cloth_space)
        total_area = self.cloth_width * self.cloth_height
        return used_area / total_area

    def _calculate_compactness(self):
        """Calculate how compactly patterns are placed"""
        if np.sum(self.cloth_space) == 0:
            return 0.0

        # Find the bounding box of the used area
        rows = np.any(self.cloth_space, axis=1)
        cols = np.any(self.cloth_space, axis=0)

        height = np.sum(rows)
        width = np.sum(cols)

        used_area = np.sum(self.cloth_space)
        bounding_area = height * width

        return used_area / bounding_area if bounding_area > 0 else 0.0

    def _get_state(self):
        """Get current state representation"""
        try:
            # Create state channels
            cloth_space = self.cloth_space.astype(np.float32)

            # Create pattern channel for next pattern
            pattern_channel = self._create_pattern_channel()

            # Create distance transform channel
            distance_channel = self._create_distance_channel(cloth_space)

            # Stack all channels
            pattern_channels = [cloth_space, pattern_channel, distance_channel]

            # Add a channel for each pattern (placeholder for future features)
            for _ in range(len(self.patterns)):
                pattern_channels.append(np.zeros_like(cloth_space))

            # Ensure correct number of channels
            state = np.stack(pattern_channels[:3 + len(self.patterns)]).astype(np.float32)

            return state

        except Exception as e:
            logger.warning(f"Error creating state: {e}")
            shape = self.cloth_space.shape
            return np.zeros((3 + len(self.patterns), shape[0], shape[1]), dtype=np.float32)

    def _create_pattern_channel(self):
        """Create a channel showing the next pattern to place"""
        pattern_channel = np.zeros_like(self.cloth_space)
        if self.available_patterns:
            next_pattern_idx = self.available_patterns[0]
            next_pattern = self.patterns[next_pattern_idx]

            # Draw the pattern
            if "contours" in next_pattern and next_pattern["contours"]:
                try:
                    cv2.fillPoly(pattern_channel, next_pattern["contours"], 1)
                except Exception:
                    # Use fallback if drawing fails
                    self._draw_fallback_pattern(pattern_channel, next_pattern)
            else:
                self._draw_fallback_pattern(pattern_channel, next_pattern)

        return pattern_channel

    def _create_distance_channel(self, cloth_space):
        """Create a distance transform channel to guide placement"""
        distance_channel = np.zeros_like(cloth_space)
        if np.sum(cloth_space) > 0:
            distance_map = cv2.distanceTransform(
                (1 - cloth_space).astype(np.uint8), cv2.DIST_L2, 5
            )
            if np.max(distance_map) > 0:
                distance_channel = distance_map / np.max(distance_map)

        return distance_channel

    def _draw_fallback_pattern(self, pattern_channel, pattern):
        """Draw a fallback pattern shape when contours are not available"""
        try:
            if "dimensions" in pattern:
                # Get dimensions
                dimensions = pattern["dimensions"]
                if hasattr(dimensions, "tolist"):
                    dimensions = dimensions.tolist()

                width, height = map(int, dimensions)
                width, height = max(5, width), max(5, height)

                # Draw a rectangle at the center
                center_x = self.cloth_width // 2
                center_y = self.cloth_height // 2
                x1, y1 = center_x - width // 2, center_y - height // 2
                x2, y2 = x1 + width, y1 + height

                cv2.rectangle(pattern_channel, (x1, y1), (x2, y2), 1, -1)
            else:
                # Draw a small square if no dimensions
                center_x = self.cloth_width // 2
                center_y = self.cloth_height // 2
                size = min(self.cloth_width, self.cloth_height) // 10
                cv2.rectangle(
                    pattern_channel,
                    (center_x - size, center_y - size),
                    (center_x + size, center_y + size),
                    1, -1
                )
        except Exception as e:
            logger.warning(f"Error creating fallback pattern: {e}")


class ManagerNetwork(nn.Module):
    """Manager network for pattern selection

    Decides which pattern to place next based on the current state.
    """

    def __init__(self, input_channels, hidden_dim, num_patterns):
        """Initialize the manager network

        Args:
            input_channels: Number of input channels in state representation
            hidden_dim: Size of hidden layer
            num_patterns: Number of patterns to choose from
        """
        super().__init__()

        # Input adapter for compatibility with saved models
        self.input_adapter = nn.Conv2d(input_channels, 4, 1, bias=False)

        # Feature extraction network
        self.conv = nn.Sequential(
            # First block
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Adaptive pooling for consistent output size
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # Decision-making network
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_patterns),
        )

    def forward(self, x):
        """Forward pass through the network"""
        # Adapt input channels
        x = self.input_adapter(x)

        # Extract features
        features = self.conv(x)

        # Flatten and pass through FC layers
        x = features.view(features.size(0), -1)
        logits = self.fc(x)

        # Return pattern selection probabilities
        return F.softmax(logits, dim=1)


class WorkerNetwork(nn.Module):
    """Worker network for pattern placement

    Decides where and how to place a selected pattern.
    """

    def __init__(self, input_channels, num_rotations, cloth_dims):
        """Initialize the worker network

        Args:
            input_channels: Number of input channels in the state
            num_rotations: Number of possible rotation angles
            cloth_dims: Dimensions of the cloth (height, width)
        """
        super().__init__()

        self.cloth_height, self.cloth_width = cloth_dims

        # Input adapter
        self.input_adapter = nn.Conv2d(input_channels, 5, 1, bias=False)

        # Encoder network
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(5, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Global pooling for rotation head
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Placement head (generates position heatmap)
        self.placement_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(size=(cloth_dims[0], cloth_dims[1])),
            nn.Conv2d(32, 1, 1),
        )

        # Rotation head (selects rotation angle)
        self.rotation_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_rotations),
        )

    def forward(self, x):
        """Forward pass through the network"""
        # Adapt input channels
        x = self.input_adapter(x)

        # Extract features
        features = self.encoder(x)

        # Get placement map (where to place the pattern)
        placement_map = self.placement_head(features)

        # Get rotation probabilities (how to rotate the pattern)
        pooled = self.global_pool(features)
        flat = pooled.view(pooled.size(0), -1)
        rotation_logits = self.rotation_head(flat)
        rotation_probs = F.softmax(rotation_logits, dim=1)

        return placement_map, rotation_probs


class HierarchicalRL:
    """Hierarchical Reinforcement Learning for pattern packing

    Uses a two-level approach:
    1. Manager network selects which pattern to place next
    2. Worker network decides where and how to place it
    """

    def __init__(self, env, device=None):
        """Initialize the hierarchical RL agent

        Args:
            env: PackingEnvironment instance
            device: Computation device (cuda or cpu)
        """
        self.env = env
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get cloth dimensions
        cloth_dims = (env.cloth_height, env.cloth_width)

        # Initialize networks
        self._initialize_networks(cloth_dims)

        # Initialize optimizers
        self.manager_optimizer = optim.Adam(self.manager.parameters(), lr=0.001)
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=0.001)

        # Training parameters
        self.buffer = []
        self.gamma = 0.99

    def _initialize_networks(self, cloth_dims):
        """Initialize neural networks for the agent"""
        # Manager network (decides which pattern to place)
        self.manager = ManagerNetwork(
            input_channels=self.env.observation_space.shape[0],
            hidden_dim=256,
            num_patterns=len(self.env.patterns),
        ).to(self.device)

        # Worker network (decides where and how to place pattern)
        self.worker = WorkerNetwork(
            input_channels=self.env.observation_space.shape[0] + 1,  # +1 for pattern channel
            num_rotations=len(self.env.rotation_angles),
            cloth_dims=cloth_dims,
        ).to(self.device)

    def select_pattern(self, probs, epsilon=0.1):
        """Select a pattern using epsilon-greedy strategy

        Args:
            probs: Probability distribution over patterns
            epsilon: Exploration rate (0-1)

        Returns:
            Selected pattern index or None if no patterns available
        """
        if not self.env.available_patterns:
            return None

        # Exploration: random pattern
        if np.random.random() < epsilon:
            return np.random.choice(self.env.available_patterns)

        # Exploitation: highest probability pattern
        available_probs = np.zeros(len(self.env.available_patterns))
        for i, idx in enumerate(self.env.available_patterns):
            if idx < len(probs):
                available_probs[i] = probs[idx]

        if len(available_probs) > 0:
            # Select best available pattern
            selected_idx = self.env.available_patterns[np.argmax(available_probs)]
            return selected_idx

        # Fallback to first available pattern
        return self.env.available_patterns[0]

    def train(self, num_episodes):
        """Train both networks

        Args:
            num_episodes: Number of episodes to train for

        Returns:
            Dict containing best state and utilization
        """
        best_utilization = 0.0
        best_state = None

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            while not done:
                # Manager selects pattern
                pattern_probs = self.manager(state_tensor)
                pattern_idx = self.select_pattern(pattern_probs.squeeze().cpu().detach().numpy())

                if pattern_idx is None:
                    # No more patterns to place
                    break

                # Create worker state (add pattern channel)
                worker_state = self._create_worker_state(state, pattern_idx)
                worker_tensor = torch.FloatTensor(worker_state).unsqueeze(0).to(self.device)

                # Worker selects position and rotation
                placement_map, rotation_probs = self.worker(worker_tensor)

                # Select best position
                pos_idx = torch.argmax(placement_map.view(-1)).item()
                pos_y, pos_x = np.unravel_index(pos_idx, (self.env.cloth_height, self.env.cloth_width))

                # Select best rotation
                rotation_idx = torch.argmax(rotation_probs).item()

                # Take action
                action = {
                    "pattern_idx": pattern_idx,
                    "position": (pos_x, pos_y),
                    "rotation": rotation_idx,
                }

                # Step environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store experience for training
                self._store_experience(state, action, reward, next_state, done)

                # Update networks if buffer is large enough
                if len(self.buffer) >= 16:
                    self._update_networks()
                    self.buffer = []

                # Update state and reward
                state = next_state
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                episode_reward += reward

                # Track best state
                current_utilization = info["utilization"]
                if current_utilization > best_utilization:
                    best_utilization = current_utilization
                    best_state = self.env.cloth_space.copy()

            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, "
                           f"Utilization: {info['utilization']:.4f}")

        return {"best_state": best_state, "best_utilization": best_utilization}

    def _create_worker_state(self, state, pattern_idx):
        """Create input state for worker network by adding pattern channel"""
        pattern_channel = np.zeros_like(self.env.cloth_space)
        pattern = self.env.patterns[pattern_idx]

        # Draw pattern contours if available
        if "contours" in pattern and pattern["contours"]:
            try:
                cv2.fillPoly(pattern_channel, pattern["contours"], 1)
            except Exception:
                # Use fallback if drawing fails
                self.env._draw_fallback_pattern(pattern_channel, pattern)
        else:
            self.env._draw_fallback_pattern(pattern_channel, pattern)

        # Add pattern channel to state
        worker_state = np.concatenate([state, pattern_channel[np.newaxis, :, :]])
        return worker_state

    def _store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        })

    def _update_networks(self):
        """Update networks using accumulated experiences"""
        # Skip if buffer is empty
        if not self.buffer:
            return

        # Prepare batch data
        states = torch.FloatTensor([x["state"] for x in self.buffer]).to(self.device)
        rewards = torch.FloatTensor([x["reward"] for x in self.buffer]).to(self.device)
        actions = [x["action"] for x in self.buffer]

        # Update manager network
        self._update_manager(states, actions, rewards)

        # Update worker network
        self._update_worker(actions, rewards)

    def _update_manager(self, states, actions, rewards):
        """Update the manager network"""
        pattern_probs = self.manager(states)
        manager_loss = 0

        for i, action in enumerate(actions):
            if action["pattern_idx"] < pattern_probs.shape[1]:
                log_prob = torch.log(pattern_probs[i, action["pattern_idx"]] + 1e-10)
                manager_loss -= log_prob * rewards[i]

        manager_loss /= len(self.buffer)

        self.manager_optimizer.zero_grad()
        manager_loss.backward()
        self.manager_optimizer.step()

    def _update_worker(self, actions, rewards):
        """Update the worker network"""
        # Prepare worker states
        worker_states = []
        for x in self.buffer:
            state = x["state"]
            pattern_idx = x["action"]["pattern_idx"]

            # Create worker state with pattern channel
            worker_state = self._create_worker_state(state, pattern_idx)
            worker_states.append(worker_state)

        worker_tensors = torch.FloatTensor(worker_states).to(self.device)

        # Get worker predictions
        placement_maps, rotation_probs = self.worker(worker_tensors)

        # Prepare targets
        placement_targets = self._get_placement_targets(actions)
        rotation_targets = self._get_rotation_targets(actions)

        # Calculate losses
        placement_loss = F.cross_entropy(
            placement_maps.view(len(self.buffer), -1), placement_targets
        )
        rotation_loss = F.cross_entropy(rotation_probs, rotation_targets)

        # Combined loss
        worker_loss = placement_loss + rotation_loss

        # Update weights
        self.worker_optimizer.zero_grad()
        worker_loss.backward()
        self.worker_optimizer.step()

    def _get_placement_targets(self, actions):
        """Get target positions from actions"""
        placement_targets = []
        for action in actions:
            pos_x, pos_y = action["position"]
            target = pos_y * self.env.cloth_width + pos_x
            placement_targets.append(target)

        return torch.LongTensor(placement_targets).to(self.device)

    def _get_rotation_targets(self, actions):
        """Get target rotations from actions"""
        rotation_targets = []
        for action in actions:
            if isinstance(action["rotation"], int):
                rotation_targets.append(action["rotation"])
            else:
                # For compatibility with both rotation index and angle value
                try:
                    rot_angle = action["rotation"]
                    rot_idx = self.env.rotation_angles.index(rot_angle)
                    rotation_targets.append(rot_idx)
                except (ValueError, TypeError):
                    # Fallback: use the first rotation
                    rotation_targets.append(0)

        return torch.LongTensor(rotation_targets).to(self.device)

    def save_model(self, model_path):
        """Save trained models"""
        torch.save(
            {
                "manager": self.manager.state_dict(),
                "worker": self.worker.state_dict(),
            },
            model_path,
        )
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """Load trained models"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.manager.load_state_dict(checkpoint["manager"], strict=False)
        self.worker.load_state_dict(checkpoint["worker"], strict=False)
        logger.info(f"Model loaded from {model_path}")

    def infer(self):
        """Run inference to find optimal pattern placement

        Returns:
            Tuple of (final_state, utilization, placement_data)
        """
        state, _ = self.env.reset()
        done = False
        total_reward = 0

        # Maximum attempts
        max_attempts = len(self.env.patterns) * 3
        step = 0
        placed_count = 0

        while not done and step < max_attempts and placed_count < len(self.env.patterns):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get pattern selection from manager
            with torch.no_grad():
                pattern_probs = self.manager(state_tensor)

            # Select pattern
            pattern_idx = self.select_pattern(pattern_probs.squeeze().cpu().numpy(), epsilon=0.1)

            if pattern_idx is None:
                break

            # Create worker state with pattern
            worker_state = self._create_worker_state(state, pattern_idx)
            worker_tensor = torch.FloatTensor(worker_state).unsqueeze(0).to(self.device)

            # Get placement suggestions from worker
            with torch.no_grad():
                placement_map, rotation_probs = self.worker(worker_tensor)

                # Get top positions to try
                placement_suggestions = self._get_top_placement_suggestions(placement_map)

                # Get top rotations to try
                rotation_suggestions = self._get_top_rotation_suggestions(rotation_probs)

            # Try different positions and rotations
            placement_success = self._try_placements(
                pattern_idx, placement_suggestions, rotation_suggestions
            )

            if placement_success:
                placed_count += 1
                total_reward += placement_success["reward"]
                state = placement_success["next_state"]
                logger.info(f"Step {step}: Placed pattern {pattern_idx} with "
                          f"utilization: {placement_success['utilization']:.4f}")
            else:
                logger.warning(f"Failed to place pattern {pattern_idx}")

            done = placed_count >= len(self.env.patterns)
            step += 1

        # Get final results
        final_utilization = self.env._calculate_utilization()
        logger.info(f"Inference complete: Utilization {final_utilization:.4f}, "
                   f"Placed {placed_count}/{len(self.env.patterns)} patterns")

        return self.env.cloth_space, final_utilization, self.env.placed_patterns

    def _get_top_placement_suggestions(self, placement_map, k=5):
        """Get top-k placement positions from heatmap"""
        flattened_map = placement_map.view(-1)
        top_values, top_indices = torch.topk(flattened_map, k=k)

        positions = []
        for idx in top_indices:
            pos_y, pos_x = np.unravel_index(idx.item(), (self.env.cloth_height, self.env.cloth_width))
            positions.append((pos_x, pos_y))

        return positions

    def _get_top_rotation_suggestions(self, rotation_probs, k=2):
        """Get top-k rotation angles"""
        top_rot_values, top_rot_indices = torch.topk(rotation_probs.squeeze(), k=k)
        return [idx.item() for idx in top_rot_indices]

    def _try_placements(self, pattern_idx, positions, rotations):
        """Try different positions and rotations until one works"""
        for position in positions:
            for rotation in rotations:
                action = {
                    "pattern_idx": pattern_idx,
                    "position": position,
                    "rotation": rotation,
                }

                next_state, reward, terminated, truncated, info = self.env.step(action)

                if info.get("success", False):
                    return {
                        "next_state": next_state,
                        "reward": reward,
                        "utilization": info["utilization"],
                    }

        return None


class PatternFittingModule:
    """Module for fitting patterns onto cloth

    Main interface for pattern fitting functionality.
    """

    def __init__(self, model_path=None):
        """Initialize the pattern fitting module

        Args:
            model_path: Path to pretrained model or None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.rl_agent = None
        self.rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        logger.info(f"Pattern Fitting Module initialized on device: {self.device}")

    def prepare_data(self, cloth_data, patterns_data):
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

    def _process_cloth_data(self, cloth_data):
        """Process cloth data for the environment"""
        processed_cloth = cloth_data.copy()

        # Use cloth mask if available
        self._process_cloth_mask(processed_cloth)

        # Ensure valid dimensions
        self._ensure_valid_cloth_dimensions(processed_cloth)

        # Scale down very large cloth
        self._scale_cloth_if_needed(processed_cloth)

        return processed_cloth

    def _process_cloth_mask(self, cloth_data):
        """Process cloth mask to extract contours and dimensions"""
        cloth_mask = cloth_data.get("cloth_mask", None)
        if cloth_mask is not None and np.sum(cloth_mask) > 0:
            try:
                contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours and len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 500:
                        cloth_data["contours"] = [largest_contour]

                        # Update dimensions
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        if w > 10 and h > 10:
                            cloth_data["dimensions"] = np.array([w, h], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Error processing cloth mask: {e}")

    def _ensure_valid_cloth_dimensions(self, cloth_data):
        """Ensure cloth has valid dimensions"""
        dimensions = cloth_data.get("dimensions", None)
        if dimensions is None or np.any(np.array(dimensions) < 10):
            dimensions = np.array([512.0, 512.0], dtype=np.float32)
            cloth_data["dimensions"] = dimensions

    def _scale_cloth_if_needed(self, cloth_data):
        """Scale down cloth if it's too large"""
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

    def _process_patterns_data(self, patterns_data, processed_cloth):
        """Process patterns data for the environment"""
        processed_patterns = []
        cloth_width, cloth_height = processed_cloth["dimensions"]
        max_size = 512
        need_scaling = cloth_width > max_size or cloth_height > max_size
        scale = max_size / max(cloth_width, cloth_height) if need_scaling else 1.0

        for pattern in patterns_data:
            try:
                processed_pattern = pattern.copy()

                # Ensure pattern has valid contours
                if "contours" not in processed_pattern or not processed_pattern["contours"]:
                    self._create_contours_from_dimensions(processed_pattern)

                # Scale patterns if cloth was scaled
                if need_scaling:
                    self._scale_pattern_contours(processed_pattern, scale)

                processed_patterns.append(processed_pattern)
            except Exception as e:
                logger.warning(f"Error processing pattern: {e}")

        return processed_patterns

    def _create_contours_from_dimensions(self, pattern):
        """Create rectangle contour from dimensions"""
        if "dimensions" in pattern:
            # Create rectangle contour from dimensions
            dims = pattern["dimensions"]
            if hasattr(dims, "tolist"):
                dims = dims.tolist()
            width, height = map(float, dims)

            # Create rectangle
            rect = np.array([
                [[0, 0]], [[width, 0]], [[width, height]], [[0, height]]
            ], dtype=np.float32)

            pattern["contours"] = [rect]

    def _scale_pattern_contours(self, pattern, scale):
        """Scale pattern contours by given factor"""
        try:
            for i, contour in enumerate(pattern["contours"]):
                pattern["contours"][i] = contour * scale
        except Exception as e:
            logger.warning(f"Error scaling pattern contours: {e}")

    def train(self, cloth_data, patterns_data, num_episodes=100):
        """Train the pattern fitting model

        Args:
            cloth_data: Dictionary with cloth properties
            patterns_data: List of pattern dictionaries
            num_episodes: Number of training episodes

        Returns:
            Training result dictionary
        """
        # Prepare data
        processed_cloth, processed_patterns = self.prepare_data(cloth_data, patterns_data)

        # Create environment
        env = PackingEnvironment(
            cloth_data=processed_cloth,
            patterns=processed_patterns,
            rotation_angles=self.rotation_angles,
        )

        # Create RL agent
        self.rl_agent = HierarchicalRL(env, self.device)

        # Train the agent
        logger.info(f"Starting training for {num_episodes} episodes")
        result = self.rl_agent.train(num_episodes)

        # Save model if path specified
        if self.model_path:
            self.rl_agent.save_model(self.model_path)

        return result

    def fit_patterns(self, cloth_data, patterns_data, visualize=False):
        """Fit patterns onto cloth

        Args:
            cloth_data: Dictionary with cloth properties
            patterns_data: List of pattern dictionaries
            visualize: Whether to enable visualization

        Returns:
            Dictionary with fitting results
        """
        # Prepare data
        processed_cloth, processed_patterns = self.prepare_data(cloth_data, patterns_data)

        # Create environment
        env = PackingEnvironment(
            cloth_data=processed_cloth,
            patterns=processed_patterns,
            rotation_angles=self.rotation_angles,
        )

        # Create or load RL agent
        if self.rl_agent is None:
            self.rl_agent = HierarchicalRL(env, self.device)

            # Load model if available
            if self.model_path and os.path.exists(self.model_path):
                self.rl_agent.load_model(self.model_path)
                logger.info(f"Loaded trained model from {self.model_path}")
            else:
                # Train briefly if no model
                logger.warning("No pretrained model available, training for 20 episodes")
                self.rl_agent.train(20)

        # Run inference
        final_state, utilization, placement_data = self.rl_agent.infer()

        # Format placement data
        placements = []
        for placement in placement_data:
            placements.append({
                "pattern_id": placement["pattern_idx"],
                "position": placement["position"],
                "rotation": placement["rotation"],
            })

        return {
            "final_state": final_state,
            "utilization": utilization,
            "placements": placements,
            "cloth_dims": (env.cloth_width, env.cloth_height),
            "patterns": processed_patterns,
        }
