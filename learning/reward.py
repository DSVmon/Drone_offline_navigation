"""
Reward computation for MAVRL-style training.

MAVRL config.yaml coefficients:
  distance_coeff: -0.000, angle_vel_coeff: -0.000, input_coeff: -0.0003
  yaw_coeff: -0.000, vert_coeff: -0.002, vel_coeff: -0.000, colli_coeff: -0.00

MAVRL uses ONLY vert_coeff + input_coeff. All others = 0.
We keep R_GOAL_COEFF because we don't have AvoidBench's built-in goal reward.

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
    if info is None:
        info = {}
    terminated = False
    reward = 0.0

    dist_to_goal = np.linalg.norm(pos - goal_point)

    # 1. Goal progress (our addition — MAVRL has this built into AvoidBench)
    prev_pos_3d = np.array([prev_pos[0], prev_pos[1], pos[2]])
    prev_dist = np.linalg.norm(prev_pos_3d - goal_point)
    progress = prev_dist - dist_to_goal
    reward += config.R_GOAL_COEFF * progress

    # 2. Action penalties (matching MAVRL config.yaml exactly)
    if action is not None and prev_action is not None:
        # input_coeff = -0.0003: penalty for action changes
        action_delta = np.linalg.norm(np.array(action) - np.array(prev_action))
        reward += config.R_INPUT_PENALTY * action_delta

        # vert_coeff = -0.002: penalty for vertical input
        reward += config.R_VERTICAL_PENALTY * abs(prev_vz_input)

    # 3. Collision → reset (MAVRL: reset_if_collide=true, no penalty)
    if collision_detected:
        info["termination_reason"] = "collision"
        terminated = True

    # 4. Stuck → reset (our addition, MAVRL has timeout instead)
    if stuck:
        reward += config.R_STUCK
        info["termination_reason"] = "stuck"
        terminated = True

    # 5. Out of bounds → reset (MAVRL has bounding_box check)
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

    # 6. Goal reached → success
    if dist_to_goal < config.GOAL_REACHED_THRESHOLD:
        reward += config.R_COMPLETION
        info["termination_reason"] = "completed_lap"
        terminated = True

    # 7. Timeout → reset (MAVRL: max_t=5.0, no penalty)
    if elapsed_time > config.EPISODE_TIMEOUT_SEC:
        info["termination_reason"] = "timeout"
        terminated = True

    return reward, terminated, info
