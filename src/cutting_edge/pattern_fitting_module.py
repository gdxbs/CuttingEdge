import logging
import os
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
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

    def __init__(self, cloth_data: Dict, patterns: List[Dict], rotation_angles: List[int]):
        self.cloth_data = cloth_data
        self.cloth_width, self.cloth_height = map(int, cloth_data["dimensions"])
        self.patterns = patterns
        self.current_state = None
        self.placed_patterns = []
        self.grid_size = 1  # Minimum unit for placement
        self.rotation_angles = rotation_angles

        # Initialize cloth space
        self.reset()

    def reset(self):
        """Reset environment to initial state"""

        self.cloth_space = np.zeros(
            (self.cloth_height, self.cloth_width), dtype=np.uint8
        )
        self.current_state = self._get_state()
        self.placed_patterns = []
        self.available_patterns = list(range(len(self.patterns)))  # Indices of available patterns

        return self.current_state

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        action: Dict containing pattern_idx, position, rotation
        """

        pattern_idx = action["pattern_idx"]
        position = action["position"]
        rotation = action["rotation"]

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

        # Try to place pattern
        success = self._place_pattern(pattern, position, rotation)

        if success:
            reward = self._calculate_reward()
            self.placed_patterns.append(
                {"pattern": pattern, "position": position, "rotation": rotation, "pattern_idx": pattern_idx}
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
        """Attempt to place pattern at given position and rotation"""

        x, y = position

        # Create pattern mask
        pattern_mask = np.zeros_like(self.cloth_space)
        rotated_contour = self._rotate_contour(pattern["contours"][0], rotation)
        translated_contour = rotated_contour + np.array([x, y])

        # Check boundaries
        if (
            np.any(translated_contour < 0)
            or np.any(translated_contour[:, 0] >= self.cloth_width)
            or np.any(translated_contour[:, 1] >= self.cloth_height)
        ):
            return False

        # Fill pattern mask
        cv2.fillPoly(pattern_mask, [translated_contour.astype(int)], 1)

        # Check overlap
        if np.any(np.logical_and(self.cloth_space, pattern_mask)):
            return False

        # Place pattern
        self.cloth_space = np.logical_or(self.cloth_space, pattern_mask).astype(
            np.uint8
        )

        return True

    def _rotate_contour(self, contour: np.ndarray, angle: float) -> np.ndarray:
        """Rotate contour points around center"""

        center = np.mean(contour, axis=0)
        angle_rad = np.radians(angle)
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

        centered_contour = contour - center
        rotated_contour = centered_contour @ rotation_matrix.T

        return rotated_contour + center

    def _calculate_reward(self) -> float:
        """Calculate reward based on multiple factors"""

        utilization = self._calculate_utilization()
        compactness = self._calculate_compactness()
        edge_distance = self._calculate_edge_distance()
        valid_placement = 1.0  # All placements are valid at this point

        # Weighted reward calculation
        reward = (0.5 * utilization + 0.3 * compactness + 0.1 * edge_distance + 0.1 * valid_placement)

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


class ManagerNetwork(nn.Module):
    """High-level network for sequence planning
    
    This network determines the optimal order in which to place patterns.
    It takes the current state (cloth space + pattern features) and outputs
    a probability distribution over patterns to place next.
    
    Architecture:
    - CNN layers for processing spatial information
    - Fully connected layers for pattern selection
    - Softmax output for pattern selection probabilities
    
    The manager network plays a critical role in the hierarchical RL approach,
    providing strategic planning while the worker network handles tactical execution.
    
    Based on the manager-worker architecture described in:
    "Planning Irregular Object Packing via Hierarchical Reinforcement Learning" (Wang et al., 2022)
    """

    def __init__(self, input_channels: int, hidden_dim: int, num_shapes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Add adaptive pooling to handle variable cloth sizes
            nn.AdaptiveAvgPool2d((16, 16))
        )
        
        # Fixed input size after adaptive pooling
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, hidden_dim),  # Fixed size regardless of input dimensions
            nn.ReLU(),
            nn.Linear(hidden_dim, num_shapes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        sequence_probs = F.softmax(self.fc(x), dim=1)
        return sequence_probs, None


class WorkerNetwork(nn.Module):
    """Low-level network for placement planning
    
    This network determines the optimal position and rotation for placing a selected pattern.
    Given a state and pattern to place, it outputs:
    1. A placement map (heatmap of position probabilities)
    2. A distribution over possible rotations
    
    Architecture:
    - CNN layers for spatial feature extraction
    - Two output heads:
        - Placement head: Uses transposed convolutions to generate a pixel-wise placement heatmap
        - Rotation head: Fully connected layers to select optimal rotation angle
    
    Based on the placement policy network described in:
    "Tree Search + Reinforcement Learning for Two-Dimensional Cutting Stock Problem
     With Complex Constraints" (Zhang et al., 2023)
    """

    def __init__(self, input_channels: int, num_rotations: int, cloth_dims: Tuple[int, int]):
        super().__init__()
        self.cloth_height, self.cloth_width = cloth_dims
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Add adaptive pooling to handle variable cloth sizes
            nn.AdaptiveAvgPool2d((16, 16))
        )

        # Placement head with proper dimensions - uses up-sampling to restore original size
        self.placement_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            # Use interpolation to get to original cloth dimensions
            nn.Upsample(size=(self.cloth_height, self.cloth_width), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 1)
        )

        # Rotation head with fixed input size after adaptive pooling
        self.rotation_head = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),  # Fixed size regardless of input dimensions
            nn.ReLU(),
            nn.Linear(256, num_rotations),
        )

    def forward(self, x):
        features = self.conv(x)
        placement_map = self.placement_head(features)
        rotation_logits = self.rotation_head(features.view(features.size(0), -1))
        rotation_probs = F.softmax(rotation_logits, dim=1)
        return placement_map, rotation_probs


class HierarchicalRL:
    """Hierarchical Reinforcement Learning for pattern packing
    
    This class implements a two-level hierarchical reinforcement learning approach
    for optimizing pattern placement on cloth materials. It consists of:
    
    1. Manager network: Decides which pattern to place next (high-level strategy)
    2. Worker network: Decides where and at what rotation to place it (low-level execution)
    
    The hierarchical structure allows for complex decision-making by separating
    the "what" (pattern selection) from the "how" (placement details). This approach
    has been shown to perform well on complex packing problems with irregular shapes.
    
    Training uses policy gradient methods with experience replay to optimize
    both networks simultaneously. Reward signals are based on material utilization,
    compactness, and other placement quality metrics.
    
    Key features:
    - Epsilon-greedy exploration for pattern selection
    - Experience buffer for batch learning
    - Progressive improvement of material utilization
    
    References:
    - "Planning Irregular Object Packing via Hierarchical Reinforcement Learning" (Wang et al., 2022)
    - "Tree Search + Reinforcement Learning for Two-Dimensional Cutting Stock Problem
       With Complex Constraints" (Zhang et al., 2023)
    """

    def __init__(self, env: PackingEnvironment, device: torch.device = None):
        self.env = env
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get cloth dimensions
        cloth_dims = (self.env.cloth_height, self.env.cloth_width)

        # Initialize networks
        self.manager = ManagerNetwork(
            input_channels=3,  # State channels: cloth space, pattern, distance
            hidden_dim=256,
            num_shapes=len(env.patterns),
        ).to(self.device)

        self.worker = WorkerNetwork(
            input_channels=4,  # State channels + selected pattern
            num_rotations=len(env.rotation_angles),
            cloth_dims=cloth_dims,
        ).to(self.device)

        # Initialize optimizers
        self.manager_optimizer = optim.Adam(self.manager.parameters(), lr=0.001)
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=0.001)

        # Experience buffer
        self.buffer = []
        self.gamma = 0.99  # Discount factor

    def train(self, num_episodes: int):
        """Train both networks through reinforcement learning
        
        Trains the manager and worker networks over multiple episodes, improving
        their pattern placement strategies through trial and error. Each episode
        involves placing all patterns onto the cloth, with experiences collected
        and used to update network weights.
        
        The training process uses policy gradient methods, with the rewards
        based on material utilization, pattern compactness, and other metrics
        that evaluate placement quality.
        
        Args:
            num_episodes: Number of training episodes to run
            
        Returns:
            Tuple of (best_state, best_utilization) achieved during training
        """
        
        best_utilization = 0.0
        best_state = None

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            # Move state to device and correct format
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            while not done:
                # Get sequence from manager
                sequence_probs, _ = self.manager(state_tensor)
                # Select pattern using epsilon-greedy strategy
                pattern_idx = self.select_pattern(sequence_probs.squeeze().cpu().detach().numpy(), state)
                
                if pattern_idx is None:
                    # No more patterns to place
                    logger.info("All patterns placed, ending episode.")
                    done = True
                    break

                # Get placement from worker
                # Create worker state by concatenating current state with pattern channel
                pattern_channel = self.env._get_pattern_channel()
                worker_state = np.concatenate([state, pattern_channel[np.newaxis, :, :]])
                worker_state_tensor = torch.FloatTensor(worker_state).unsqueeze(0).to(self.device)
                
                placement_map, rotation_probs = self.worker(worker_state_tensor)

                # Select position and rotation
                # For position, find the pixel with highest probability
                pos_idx = torch.argmax(placement_map.view(-1)).item()
                pos_y, pos_x = np.unravel_index(
                    pos_idx, 
                    (self.env.cloth_height, self.env.cloth_width)
                )
                
                # For rotation, select the rotation angle with highest probability
                rotation_idx = torch.argmax(rotation_probs).item()
                rotation = self.env.rotation_angles[rotation_idx]

                # Take action
                action = {
                    "pattern_idx": pattern_idx,
                    "position": (pos_x, pos_y),
                    "rotation": rotation,
                }

                next_state, reward, done, info = self.env.step(action)

                # Store experience for learning
                self.buffer.append(
                    {
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "done": done,
                        "sequence_probs": sequence_probs.squeeze().cpu().detach().numpy(),
                        "placement_map": placement_map.squeeze().cpu().detach().numpy(),
                        "rotation_probs": rotation_probs.squeeze().cpu().detach().numpy()
                    }
                )

                # Update networks if buffer has enough samples
                if len(self.buffer) >= 32:  # Batch size
                    self._update_networks()
                    self.buffer = []

                # Update state and running reward
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

    def select_pattern(self, sequence_probs, state, epsilon=0.1):
        """Select a pattern, implementing epsilon-greedy exploration
        
        Uses an epsilon-greedy policy to balance exploration and exploitation:
        - With probability epsilon: randomly select from available patterns (exploration)
        - With probability (1-epsilon): select pattern with highest probability (exploitation)
        
        This approach ensures the agent explores different pattern sequences during training,
        while gradually focusing on the most promising options as it learns.
        
        Args:
            sequence_probs: Probability distribution over patterns from manager network
            state: Current environment state
            epsilon: Exploration rate (0.0 to 1.0)
            
        Returns:
            Selected pattern index or None if no patterns available
        """
        if not self.env.available_patterns:
            return None  # No patterns available
            
        if np.random.random() < epsilon:
            # Explore: choose a random available pattern
            return np.random.choice(self.env.available_patterns)
        else:
            # Exploit: choose the pattern with the highest probability
            valid_actions = self.env.available_patterns
            if not valid_actions:
                return None  # No valid actions
            
            # Filter probabilities for valid actions
            valid_probs = sequence_probs[valid_actions]
            
            if len(valid_probs) > 0:
                best_action = valid_actions[np.argmax(valid_probs)]
                return best_action
            else:
                return np.random.choice(valid_actions)

    def _update_networks(self):
        """Update both networks using stored experiences
        
        Performs policy gradient updates for both manager and worker networks
        using the experiences collected during interaction with the environment.
        
        The update process:
        1. For the manager: Updates pattern selection policy based on rewards received
        2. For the worker: Updates position and rotation selection based on successful placements
        
        This method implements a form of REINFORCE algorithm (policy gradient),
        where the networks learn to maximize expected rewards through gradient ascent.
        Parameters are updated in the direction that increases the log probability
        of actions that led to high rewards.
        """
        # Sample batch
        batch = self.buffer  # Use all experiences in buffer

        # Prepare batch data
        states = torch.FloatTensor([x["state"] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x["reward"] for x in batch]).to(self.device)
        actions = [x["action"] for x in batch]

        # Update manager
        sequence_probs, _ = self.manager(states)
        
        # Calculate manager loss
        manager_loss = 0
        for i, action in enumerate(actions):
            # Use log probability of the chosen action multiplied by the reward
            if action["pattern_idx"] < sequence_probs.shape[1]:
                log_prob = torch.log(sequence_probs[i, action["pattern_idx"]] + 1e-10)  # Add small value to prevent log(0)
                manager_loss -= log_prob * rewards[i]  # Gradient ascent

        manager_loss /= len(batch)  # Normalize by batch size

        self.manager_optimizer.zero_grad()
        manager_loss.backward()
        self.manager_optimizer.step()

        # Update worker by creating worker states
        worker_states = []
        for x in batch:
            state = x["state"]
            pattern_idx = x["action"]["pattern_idx"]
            pattern_mask = np.zeros_like(self.env.cloth_space)
            if pattern_idx < len(self.env.patterns):
                cv2.fillPoly(pattern_mask, self.env.patterns[pattern_idx]["contours"], 1)
            worker_state = np.concatenate([state, pattern_mask[np.newaxis, :, :]])
            worker_states.append(worker_state)
            
        worker_states_tensor = torch.FloatTensor(worker_states).to(self.device)
        placement_maps, rotation_probs = self.worker(worker_states_tensor)

        # Calculate worker losses
        placement_targets = []
        for action in actions:
            pos_x, pos_y = action["position"]
            target = pos_y * self.env.cloth_width + pos_x
            placement_targets.append(target)
            
        placement_targets_tensor = torch.LongTensor(placement_targets).to(self.device)
        
        # Calculate placement loss
        placement_loss = F.cross_entropy(
            placement_maps.view(len(batch), -1),
            placement_targets_tensor
        )

        # Calculate rotation loss
        rotation_targets = [self.env.rotation_angles.index(x["action"]["rotation"]) for x in batch]
        rotation_targets_tensor = torch.LongTensor(rotation_targets).to(self.device)
        
        rotation_loss = F.cross_entropy(rotation_probs, rotation_targets_tensor)

        # Combine worker losses
        worker_loss = placement_loss + rotation_loss

        self.worker_optimizer.zero_grad()
        worker_loss.backward()
        self.worker_optimizer.step()

    def save_model(self, model_path: str):
        """Save trained models"""
        torch.save({
            "manager": self.manager.state_dict(),
            "worker": self.worker.state_dict()
        }, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load trained models"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.manager.load_state_dict(checkpoint["manager"])
        self.worker.load_state_dict(checkpoint["worker"])
        logger.info(f"Model loaded from {model_path}")

    def infer(self, visualize=False):
        """Run inference to find optimal pattern placement
        
        Uses the trained manager and worker networks to find an optimal
        pattern placement solution without exploration (pure exploitation).
        
        The inference process:
        1. Reset environment to initial state
        2. For each step:
           - Manager selects next pattern to place
           - Worker determines optimal position and rotation
           - Pattern is placed on the cloth
        3. Continue until all patterns are placed or no valid placements remain
        
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
            plt.imshow(self.env.cloth_space, cmap='gray')
            plt.title("Initial cloth")
            plt.savefig(os.path.join(output_dir, "initial_cloth.png"))
            plt.close()
            
        step = 0
        while not done:
            # Move state to device
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get pattern selection from manager
            with torch.no_grad():
                sequence_probs, _ = self.manager(state_tensor)
            
            # Select best pattern without exploration
            pattern_idx = self.select_pattern(sequence_probs.squeeze().cpu().numpy(), state, epsilon=0)
            
            if pattern_idx is None:
                # No more patterns to place
                break
                
            # Get placement from worker
            pattern_channel = self.env._get_pattern_channel()
            worker_state = np.concatenate([state, pattern_channel[np.newaxis, :, :]])
            worker_state_tensor = torch.FloatTensor(worker_state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                placement_map, rotation_probs = self.worker(worker_state_tensor)
            
            # Select position and rotation
            pos_idx = torch.argmax(placement_map.view(-1)).item()
            pos_y, pos_x = np.unravel_index(
                pos_idx, 
                (self.env.cloth_height, self.env.cloth_width)
            )
            
            rotation_idx = torch.argmax(rotation_probs).item()
            rotation = self.env.rotation_angles[rotation_idx]
            
            # Take action
            action = {
                "pattern_idx": pattern_idx,
                "position": (pos_x, pos_y),
                "rotation": rotation,
            }
            
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # Visualize step if requested
            if visualize:
                plt.figure(figsize=(10, 10))
                plt.imshow(self.env.cloth_space, cmap='gray')
                plt.title(f"Step {step}: Pattern {pattern_idx} placed")
                plt.savefig(os.path.join(output_dir, f"step_{step}_cloth.png"))
                plt.close()
                
            state = next_state
            step += 1
            
            # Log progress
            logger.info(f"Step {step}: Placed pattern {pattern_idx} at {(pos_x, pos_y)} with rotation {rotation}°")
            logger.info(f"Current utilization: {info['utilization']:.4f}")
            
        # Final visualization
        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.env.cloth_space, cmap='gray')
            plt.title(f"Final placement: Utilization {info['utilization']:.4f}")
            plt.savefig(os.path.join(output_dir, "final_cloth.png"))
            plt.close()
            
        logger.info(f"Inference complete: Utilization {info['utilization']:.4f}, Reward {total_reward:.2f}")
        return self.env.cloth_space, info['utilization'], self.env.placed_patterns


class PatternFittingModule:
    """Module for fitting patterns onto cloth using Hierarchical RL
    
    This module implements a pattern cutting and fitting system that uses
    Hierarchical Reinforcement Learning to optimize pattern placement on cloth material.
    It combines a high-level manager network for pattern selection sequencing with a
    low-level worker network for precise positioning and rotation.
    
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
        self.rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # Possible rotation angles
        
        logger.info(f"Pattern Fitting Module initialized on device: {self.device}")

    def prepare_data(self, cloth_data: Dict, patterns_data: List[Dict]) -> Tuple[Dict, List[Dict]]:
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
            logger.info(f"Scaled cloth from {(cloth_width, cloth_height)} to {processed_cloth['dimensions']}")
            
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
                logger.warning(f"Pattern missing contours, skipping")
                continue
            
            # Apply same scale to patterns if cloth was scaled
            if cloth_width > max_size or cloth_height > max_size:
                for i, contour in enumerate(processed_pattern["contours"]):
                    processed_pattern["contours"][i] = contour * scale
                
                # Scale dimensions if they exist
                if "dimensions" in processed_pattern:
                    processed_pattern["dimensions"] = tuple(d * scale for d in processed_pattern["dimensions"])
                
            processed_patterns.append(processed_pattern)
            
        logger.info(f"Prepared {len(processed_patterns)} patterns for fitting")
        return processed_cloth, processed_patterns

    def train(self, cloth_data: Dict, patterns_data: List[Dict], num_episodes: int = 100) -> Dict:
        """Train the pattern fitting model
        
        Args:
            cloth_data: Dictionary containing cloth properties
            patterns_data: List of dictionaries containing pattern properties
            num_episodes: Number of training episodes
            
        Returns:
            Dictionary with training results
        """
        # Prepare data for environment
        processed_cloth, processed_patterns = self.prepare_data(cloth_data, patterns_data)
        
        # Create packing environment
        env = PackingEnvironment(
            cloth_data=processed_cloth, 
            patterns=processed_patterns,
            rotation_angles=self.rotation_angles
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
            "training_episodes": num_episodes
        }
    
    def fit_patterns(self, cloth_data: Dict, patterns_data: List[Dict], visualize: bool = False) -> Dict:
        """Fit patterns onto cloth using trained model
        
        Args:
            cloth_data: Dictionary containing cloth properties
            patterns_data: List of dictionaries containing pattern properties
            visualize: Whether to generate visualization images
            
        Returns:
            Dictionary with fitting results
        """
        # Prepare data for environment
        processed_cloth, processed_patterns = self.prepare_data(cloth_data, patterns_data)
        
        # Create packing environment
        env = PackingEnvironment(
            cloth_data=processed_cloth, 
            patterns=processed_patterns,
            rotation_angles=self.rotation_angles
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
                logger.warning("No pretrained model available, training for 20 episodes")
                self.rl_agent.train(20)
        
        # Run inference to get optimal pattern placement
        final_state, utilization, placement_data = self.rl_agent.infer(visualize)
        
        # Convert placement data to more user-friendly format
        placements = []
        for placement in placement_data:
            placements.append({
                "pattern_id": placement["pattern_idx"],
                "position": placement["position"],
                "rotation": placement["rotation"]
            })
        
        return {
            "final_state": final_state,  # The cloth space with patterns placed
            "utilization": utilization,  # Material utilization percentage
            "placements": placements,    # Details of each pattern placement
            "cloth_dims": (env.cloth_width, env.cloth_height)
        }
    
    def visualize_result(self, result: Dict, cloth_image=None, save_path: str = None):
        """Visualize pattern fitting result
        
        Args:
            result: Result dictionary from fit_patterns
            cloth_image: Original cloth image (optional)
            save_path: Path to save visualization (optional)
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as patches
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create colormap for different patterns - distinct colors for each pattern
        colors = ['gray', 'red', 'blue', 'green', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'yellow']
        cmap = ListedColormap(['white'] + colors[:len(result["placements"])])
        
        # Create a visualization canvas
        cloth_width, cloth_height = result["cloth_dims"]
        canvas = np.zeros((cloth_height, cloth_width), dtype=int)
        
        # Plot the original cloth if provided
        if cloth_image is not None:
            plt.subplot(1, 2, 1)
            plt.imshow(cloth_image)
            plt.title("Original Cloth")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
        
        # Fill the canvas with pattern placements
        if "placements" in result and "patterns" in result:
            for i, placement in enumerate(result["placements"]):
                pattern_idx = placement["pattern_id"]
                position = placement["position"]
                rotation = placement["rotation"]
                
                # Label each pattern with a unique index (i+1) for color mapping
                pattern_mask = np.zeros_like(canvas)
                
                # Get pattern contours
                if pattern_idx < len(result["patterns"]) and "contours" in result["patterns"][pattern_idx]:
                    contours = result["patterns"][pattern_idx]["contours"]
                    
                    # Rotate and translate contours
                    for contour in contours:
                        center = np.mean(contour, axis=0)
                        angle_rad = np.radians(rotation)
                        rotation_matrix = np.array([
                            [np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad), np.cos(angle_rad)],
                        ])
                        centered_contour = contour - center
                        rotated_contour = centered_contour @ rotation_matrix.T
                        translated_contour = rotated_contour + center + np.array(position)
                        
                        # Fill the contour on the mask
                        cv2.fillPoly(pattern_mask, [translated_contour.astype(int)], i+1)
                    
                    # Add the pattern to the canvas
                    # Only update zeros (empty space) to avoid overwriting already placed patterns
                    canvas = np.where(canvas == 0, pattern_mask, canvas)
        
        # Display the final result
        plt.imshow(canvas, cmap=cmap, interpolation='nearest')
        plt.title(f"Pattern Placement (Utilization: {result['utilization']:.1%})")
        plt.axis('off')
        
        # Add a legend for each pattern
        handles = []
        for i, placement in enumerate(result["placements"]):
            pattern_idx = placement["pattern_id"]
            handles.append(patches.Patch(color=colors[i], label=f"Pattern {pattern_idx}"))
        
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        # Show the plot
        plt.show()
        plt.close()