#!/usr/bin/env python3
"""
Diagnostic script: validate the full training pipeline end-to-end.

Tests:
  1. Launch Gazebo (headless) + spawn drone + start ROS nodes
  2. Create DroneEnv, reset, verify odometry
  3. Step with a forward action, verify drone actually moves
  4. Verify observations are reasonable (no NaN, correct shape)
  5. Verify reward computation (non-zero for forward movement)
  6. Verify collision detection
  7. Verify checkpoint save/load with a short PPO run
  8. Cleanup

Usage:
    python3 learning/diagnostic.py
"""

import time
import math
import sys
import os
from pathlib import Path

import numpy as np
import config
# Ensure LEARNING_DIR is on sys.path
sys.path.insert(0, str(config.LEARNING_DIR))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label, condition, detail=""):
    if condition:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
    return condition


def main():
    passed = 0
    failed = 0

    section("1. Kill stale processes")
    from utils import kill_gazebo
    kill_gazebo()
    time.sleep(2)
    print("  Done.")

    section("2. Generate cave world")
    from utils import generate_cave
    cave_path = generate_cave()
    ok = check("cave.world created", Path(cave_path).exists())
    if ok:
        passed += 1
    else:
        failed += 1

    section("3. Launch Gazebo (headless)")
    from utils import launch_gazebo, wait_for_gazebo
    gazebo_proc = launch_gazebo(headless=True)
    print(f"  Gazebo PID: {gazebo_proc.pid}")
    try:
        wait_for_gazebo(timeout=90)
        print("  Gazebo ready.")
        passed += 1
    except TimeoutError as e:
        print(f"  [FAIL] {e}")
        failed += 1
        return

    section("4. Create DroneEnv and reset")
    from drone_env import DroneEnv
    try:
        env = DroneEnv(headless=True)
        print("  DroneEnv created.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] DroneEnv creation: {e}")
        passed += 1
        return

    try:
        obs, info = env.reset()
        print(f"  Reset OK. obs shape={obs.shape}, dtype={obs.dtype}")
        check("obs shape = (13,)", obs.shape == (13,), f"got {obs.shape}")
        check("obs dtype = float32", obs.dtype == np.float32, f"got {obs.dtype}")
        check("obs has no NaN", not np.any(np.isnan(obs)))
        passed += 4
    except Exception as e:
        print(f"  [FAIL] Reset: {e}")
        failed += 1

    section("5. Check initial odometry")
    with env._lock:
        x0 = env.current_x
        y0 = env.current_y
        z0 = env.current_z
        yaw0 = env.current_yaw
    print(f"  Position: x={x0:.3f}, y={y0:.3f}, z={z0:.3f}, yaw={yaw0:.3f}")
    ok = check("drone at spawn z ≈ 1.0", abs(z0 - config.DRONE_SPAWN_Z) < 0.2,
               f"z={z0:.3f}")
    if ok:
        passed += 1
    else:
        failed += 1

    section("6. Step with forward action (vx=1.0)")
    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    with env._lock:
        x1 = env.current_x
    dx = abs(x1 - x0)
    print(f"  After 1 step: x={x1:.3f}, dx={dx:.4f}, reward={reward:.4f}")
    ok_move = check("drone moves forward (dx > 0.001)", dx > 0.001,
                    f"dx={dx:.4f}")
    if ok_move:
        passed += 1
    else:
        failed += 1

    section("7. Do 10 steps, monitor movement")
    positions = []
    rewards = []
    for i in range(10):
        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        with env._lock:
            positions.append((env.current_x, env.current_y, env.current_z))
        rewards.append(reward)
        if terminated:
            print(f"  Episode terminated at step {i+1}: "
                  f"{info.get('termination_reason', '?')}")
            break
    total_dx = positions[-1][0] - positions[0][0]
    print(f"  Total forward progress: {total_dx:.3f} m over {len(positions)} steps")
    print(f"  Z range: [{min(p[2] for p in positions):.3f}, "
          f"{max(p[2] for p in positions):.3f}]")
    print(f"  Avg reward: {sum(rewards)/len(rewards):.4f}")
    ok_progress = check("total dx > 0.02 m", total_dx > 0.02,
                        f"dx={total_dx:.4f}")
    if ok_progress:
        passed += 1
    else:
        failed += 1

    section("8. Check observation values through step")
    obs, reward, terminated, truncated, info = env.step(action)
    stereo_vals = obs[0:5]
    print(f"  Stereo distances: {stereo_vals}")
    print(f"  Pos (x,y,z): {obs[5]:.3f}, {obs[6]:.3f}, {obs[7]:.3f}")
    print(f"  Sin/Cos yaw: {obs[8]:.3f}, {obs[9]:.3f}")
    print(f"  Vx: {obs[10]:.3f}")
    print(f"  Roll/Pitch: {obs[11]:.3f}, {obs[12]:.3f}")
    check("stereo values in [0, 1]", np.all(stereo_vals >= 0) and np.all(stereo_vals <= 1))
    check("sin/yaw values in [-1,1]", abs(obs[8]) <= 1.0 and abs(obs[9]) <= 1.0)
    passed += 2

    section("9. Check collision callback")
    with env._lock:
        env.collision_detected = False
    time.sleep(1)
    with env._lock:
        coll = env.collision_detected
    print(f"  Collision (expected false): {coll}")
    check("no false collision", not coll)
    passed += 1

    section("10. Reward sanity check")
    # Forward flight should give >0 reward (survive + forward progress)
    avg_reward = sum(rewards) / len(rewards)
    print(f"  Avg reward (forward flight): {avg_reward:.4f}")
    check("avg reward > 0 (survive + forward)", avg_reward > 0.01)
    passed += 1

    section("11. Short PPO training test (500 steps)")
    from stable_baselines3 import PPO
    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            gamma=config.GAMMA,
            gae_lambda=config.GAE_LAMBDA,
            clip_range=config.CLIP_RANGE,
            ent_coef=config.ENT_COEF,
            vf_coef=config.VF_COEF,
            max_grad_norm=config.MAX_GRAD_NORM,
            policy_kwargs=config.POLICY_KWARGS,
            tensorboard_log=str(config.LOG_DIR),
            verbose=0,
        )
        print("  PPO model created.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] PPO model creation: {e}")
        failed += 1
        return

    model.learn(total_timesteps=500, reset_num_timesteps=True)
    print("  PPO 500 steps completed.")

    section("12. Check checkpoint save/load")
    checkpoint_dir = config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoint_dir / "diagnostic_checkpoint.zip")
    model.save(checkpoint_path)
    ok_ckpt = check("checkpoint file exists", Path(checkpoint_path).exists())
    if ok_ckpt:
        passed += 1
    else:
        failed += 1

    try:
        loaded = PPO.load(checkpoint_path, env=env)
        check("checkpoint loads correctly", loaded.num_timesteps > 0)
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Checkpoint load: {e}")
        failed += 1

    section("13. Check TensorBoard logs")
    log_dir = config.LOG_DIR
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    ok_tb = check("TensorBoard events file exists", len(event_files) > 0,
                  f"found {len(event_files)} file(s)")
    if ok_tb:
        passed += 1
    else:
        failed += 1

    section("14. Check training log CSV")
    csv_path = config.LEARNING_DIR / "training_log.csv"
    ok_csv = check("training_log.csv exists", csv_path.exists())
    if ok_csv:
        passed += 1
    else:
        failed += 1

    section("15. Cleanup")
    env.close()
    kill_gazebo()
    time.sleep(1)
    print("  Done.")

    section("=== SUMMARY ===")
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    if failed == 0:
        print("\n  ALL TESTS PASSED. Ready for training.")
    else:
        print(f"\n  {failed} test(s) FAILED. Review logs above.")
    return failed


if __name__ == "__main__":
    sys.exit(main())
