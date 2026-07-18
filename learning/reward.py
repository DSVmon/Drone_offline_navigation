"""
Reward computation for MAVRL-style training.

Note: drone_env.py now has _compute_reward inline.
This module is kept for reference and debugging.
"""

import math
import numpy as np
import config


def compute_reward(
    pos,
    prev_pos,
    goal_point,
    vel_world,
    yaw,
    stereo_distances,
    collision_detected,
    stuck,
    elapsed_time,
    info=None,
    action=None,
    prev_action=None,
    prev_angular_vel=0.0,
    prev_vz_input=0.0,
):
    """
    MAVRL-style reward computation.

    Args:
        pos: current position [x, y, z]
        prev_pos: previous position [x, y]
        goal_point: goal position [x, y, z]
        vel_world: world-frame velocity [vx, vy, vz]
        yaw: current yaw
        stereo_distances: [left, center, right, top, bottom]
        collision_detected: bool
        stuck: bool
        elapsed_time: float
        info: dict to update

    Returns:
        reward: float
        terminated: bool
        info: dict
    """
    if info is None:
        info = {}
    terminated = False
    reward = 0.0

    # 1. Goal progress (matching MAVRL: distance_coeff × (-Δdistance))
    dist_to_goal = np.linalg.norm(pos - goal_point)
    prev_pos_3d = np.array([prev_pos[0], prev_pos[1], pos[2]])
    prev_dist = np.linalg.norm(prev_pos_3d - goal_point)
    progress = prev_dist - dist_to_goal
    reward += config.R_GOAL_COEFF * progress

    # 2. MAVRL-style action penalties (smooth flight)
    if action is not None and prev_action is not None:
        # Angular velocity penalty
        reward += config.R_ANGULAR_PENALTY * abs(prev_angular_vel)
        
        # Input change penalty (action smoothness)
        action_delta = np.linalg.norm(np.array(action) - np.array(prev_action))
        reward += config.R_INPUT_PENALTY * action_delta
        
        # Yaw rate penalty
        reward += config.R_YAW_PENALTY * abs(prev_angular_vel)
        
        # Vertical input penalty
        reward += config.R_VERTICAL_PENALTY * abs(prev_vz_input)

    # 3. Collision (terminal) — MAVRL: reset_if_collide=true, no penalty
    if collision_detected:
        info["termination_reason"] = "collision"
        terminated = True

    # 4. Stuck (terminal)
    if stuck:
        reward += config.R_STUCK
        info["termination_reason"] = "stuck"
        terminated = True

    # 8. Out of bounds (terminal)
    out_of_bounds = (
        pos[2] < config.DRONE_MIN_Z
        or pos[2] > config.DRONE_MAX_Z
        or abs(pos[0]) > config.BOUNDS_XY
        or abs(pos[1]) > config.BOUNDS_XY
    )
    if out_of_bounds:
        reward += config.R_OUT_OF_BOUNDS
        info["termination_reason"] = "out_of_bounds"
        terminated = True

    # 9. Goal reached (terminal, success)
    if dist_to_goal < config.GOAL_REACHED_THRESHOLD:
        reward += config.R_COMPLETION
        info["termination_reason"] = "completed_lap"
        terminated = True

    # 10. Timeout
    if elapsed_time > config.EPISODE_TIMEOUT_SEC:
        reward += config.R_TIMEOUT
        info["termination_reason"] = "timeout"

    return reward, terminated, info
