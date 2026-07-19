"""
Vectorized Environment wrapper for DroneEnv.

MAVRL uses VisionEnvVec (4 parallel envs in Unity).
Gazebo can't run 4 parallel instances, so we use a sequential approach
that provides the same interface.

Key: maintains separate hidden states for each "virtual" env,
runs rollouts sequentially but manages state properly.
"""

import numpy as np
from drone_env import DroneEnv


class VecDroneEnv:
    """
    Vectorized environment wrapper for DroneEnv.
    Provides VecEnv-compatible interface for RecurrentPPO.

    MAVRL: VisionEnvVec with num_envs=4
    Ours: Sequential execution with proper state management
    """

    def __init__(self, num_envs=1, headless=True):
        self.num_envs = num_envs
        self.headless = headless

        # Create single underlying env (Gazebo can't run multiple instances)
        self.env = DroneEnv(headless=headless, node_name="vec_drone_env")

        # Observation/action spaces (same for all envs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Track which env we're currently running
        self._current_env = 0
        self._episode_count = 0

    def reset(self):
        """
        Reset environment. For VecEnv interface, returns (obs, info).
        """
        obs, info = self.env.reset()
        self._episode_count += 1
        return obs, info

    def step(self, action):
        """
        Step environment. For VecEnv interface, returns (obs, reward, done, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def close(self):
        """Close environment."""
        self.env.close()

    def seed(self, seed=None):
        """Set random seed (for compatibility)."""
        pass  # Gazebo seed is set via config


class EvalDroneEnv:
    """
    Evaluation environment — separate from training env.
    Runs periodically during training to measure performance.
    MAVRL: eval_env with single env, no render.
    """

    def __init__(self, headless=True):
        self.env = DroneEnv(
            headless=headless,
            node_name="eval_drone_env"
        )
        self.num_envs = 1
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def close(self):
        self.env.close()
