import math
import config


def compute_reward(
    distances, prev_distances,
    current_x, current_y, current_z,
    prev_x, prev_y,
    odom_vx, odom_vy, odom_vz,
    yaw,
    collision_detected, collision_type,
    elapsed_time,
    entrance_heading,
    completed_lap,
    stuck,
    info
):
    terminated = False
    reward = 0.0

    left, center, right, top, bottom = distances
    p_left, p_center, p_right, p_top, p_bottom = prev_distances

    # --- 1. Базовая награда за выживание ---
    reward += config.R_SURVIVE

    # --- 2. Награда за скорость вперёд ---
    reward += config.R_SPEED_COEFF * odom_vx

    # --- 3. Штраф за близость (5 каналов) ---
    for d in distances:
        if d < config.R_PROXIMITY_THRESHOLD:
            reward += config.R_PROXIMITY_COEFF * (config.R_PROXIMITY_THRESHOLD - d) ** 2

    # --- 4. Штраф за скорость сближения (5 каналов) ---
    for d, p in zip(distances, prev_distances):
        if d < config.R_PROXIMITY_THRESHOLD and p - d > config.R_APPROACH_THRESHOLD:
            reward += config.R_APPROACH_COEFF * (p - d)

    # --- 5. Штраф за открытое пространство (вне пещеры) ---
    if left > 8.0 and right > 8.0:
        reward += config.R_OPEN_SPACE_PENALTY

    # --- 6. Горизонтальное центрирование ---
    if left < config.R_CENTERING_THRESHOLD and right < config.R_CENTERING_THRESHOLD:
        reward += config.R_CENTERING_COEFF * abs(left - right)

    # --- 7. Вертикальное центрирование ---
    if top < config.R_CENTERING_THRESHOLD and bottom < config.R_CENTERING_THRESHOLD:
        reward += config.R_CENTERING_COEFF * abs(top - bottom)

    # --- 8. Антиципация (упреждающий манёвр по высоте) ---
    if p_top > top and top < config.R_PROXIMITY_THRESHOLD:
        reward += config.R_ANTICIPATION_COEFF * (-odom_vz)
    if p_bottom > bottom and bottom < config.R_PROXIMITY_THRESHOLD:
        reward += config.R_ANTICIPATION_COEFF * odom_vz

    # --- 9. Штраф за болтанку по высоте ---
    if top > config.R_VERTICAL_BOUNCE_THRESHOLD and bottom > config.R_VERTICAL_BOUNCE_THRESHOLD:
        reward += config.R_VERTICAL_BOUNCE_COEFF * abs(odom_vz)

    # --- 10. Награда за прохождение препятствия ---
    if p_top < 0.8 and top > 1.5:
        reward += config.R_PASSAGE_REWARD
    if p_bottom < 0.8 and bottom > 1.5:
        reward += config.R_PASSAGE_REWARD

    # --- 11. Столкновение ---
    if collision_detected:
        reward += config.R_COLLISION
        info["termination_reason"] = f"collision_{collision_type}"
        terminated = True

    # --- 12. Застревание ---
    if stuck:
        reward += config.R_STUCK
        info["termination_reason"] = "stuck"
        terminated = True

    # --- 13. Полный цикл ---
    if completed_lap:
        reward += config.R_COMPLETION
        info["termination_reason"] = "completed_lap"
        terminated = True

    # --- 14. Выход за границы ---
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

    # --- 15. Таймаут ---
    if elapsed_time > config.EPISODE_TIMEOUT_SEC:
        reward += config.R_TIMEOUT
        info["termination_reason"] = "timeout"
        terminated = True

    return reward, terminated, info
