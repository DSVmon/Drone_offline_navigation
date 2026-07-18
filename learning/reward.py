import math
import config


def compute_reward(
    distances,
    current_x, current_y, current_z,
    prev_x, prev_y,
    odom_vx, odom_vy, odom_vz,
    yaw,
    collision_detected, collision_type,
    elapsed_time,
    entrance_heading,
    completed_lap,
    stuck,
    info,
    reward_overrides=None,
    prev_yaw=0.0,
    prev_distances=None,
    flight_history=None,
    current_path=None,
):
    terminated = False
    reward = 0.0

    # Apply curriculum overrides
    progress_coeff = config.R_PROGRESS_COEFF
    if reward_overrides:
        progress_coeff = reward_overrides.get("progress_coeff", progress_coeff)

    left, center, right, top, bottom = distances

    # --- 1. PROGRESS: fly forward (KEY SIGNAL) ---
    if entrance_heading is not None:
        progress_x = current_x - prev_x
        progress_y = current_y - prev_y
        progress_along_cave = progress_x * entrance_heading[0] + progress_y * entrance_heading[1]
        reward += progress_coeff * max(0.0, progress_along_cave)

    # --- 2. CENTERING: fly in the middle (HORIZONTAL) ---
    if left < 4.0 and right < 4.0:
        balance = 1.0 - abs(left - right) / max(left + right, 0.1)
        reward += 0.15 * balance

    # --- 3. CENTERING: fly in the middle (VERTICAL) ---
    if top < 4.0 and bottom < 4.0:
        balance = 1.0 - abs(top - bottom) / max(top + bottom, 0.1)
        reward += 0.15 * balance

    # --- 4. ALTITUDE: stay at correct height ---
    target_z = 1.75
    z_error = abs(current_z - target_z)
    if z_error < 0.5:
        reward += 0.10
    elif z_error < 1.0:
        reward += 0.03
    elif current_z < 0.8:
        reward -= 0.30
    elif current_z > 2.8:
        reward -= 0.10

    # --- 5. COLLISION (real + proximity + stereo) ---
    # Multiple collision detection methods:
    # 1. Real contact sensor (Gazebo bumper)
    # 2. Proximity: any stereo channel < 0.15m
    # 3. Very close: any stereo channel < 0.10m (definite collision)
    proximity_collision = any(d < 0.15 for d in distances)
    definite_collision = any(d < 0.10 for d in distances)
    if collision_detected or proximity_collision or definite_collision:
        reward += config.R_COLLISION
        if definite_collision:
            info["termination_reason"] = "collision_definite"
        elif proximity_collision:
            info["termination_reason"] = "collision_proximity"
        else:
            info["termination_reason"] = f"collision_{collision_type}"
        terminated = True

    # --- 6. STUCK (terminal) ---
    if stuck:
        reward += config.R_STUCK
        info["termination_reason"] = "stuck"
        terminated = True

    # --- 7. OUT OF BOUNDS (terminal) ---
    out_of_bounds = (
        current_z < config.DRONE_MIN_Z
        or current_z > config.DRONE_MAX_Z
        or abs(yaw) > config.DRONE_MAX_YAW
        or abs(current_x) > config.BOUNDS_XY
        or abs(current_y) > config.BOUNDS_XY
    )
    if out_of_bounds:
        reward += config.R_OUT_OF_BOUNDS
        info["termination_reason"] = "out_of_bounds"
        terminated = True

    # --- 8. COMPLETION (bonus) ---
    if completed_lap:
        reward += config.R_COMPLETION
        info["termination_reason"] = "completed_lap"
        terminated = True

    # --- 9. TIMEOUT (truncated, not terminated) ---
    if elapsed_time > config.EPISODE_TIMEOUT_SEC:
        reward += config.R_TIMEOUT
        info["termination_reason"] = "timeout"

    return reward, terminated, info
