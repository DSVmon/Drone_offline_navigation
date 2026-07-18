#!/usr/bin/env python3
"""
Training script for MAVRL-style drone navigation.

Pipeline:
    Stage A: Initial PPO (without obstacles, straight cave)
    Stage B: Collect depth data
    Stage C: Train VAE + LSTM
    Stage D: Retrain PPO with frozen encoder

Usage:
    python3 learning/train.py                          # Full pipeline
    python3 learning/train.py --stage a                # Only stage A
    python3 learning/train.py --resume <checkpoint>    # Resume from checkpoint
    python3 learning/train.py --no-bc                  # PPO from scratch
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
from drone_env import DroneEnv


def make_env(headless=True, node_name="drone_env_node"):
    return DroneEnv(headless=headless, node_name=node_name)


def lr_schedule(progress_remaining):
    """Linear LR decay from LEARNING_RATE to LEARNING_RATE_END.
    Matching MAVRL: learning_rate_schedule(progress_remaining)."""
    return config.LEARNING_RATE_END + (config.LEARNING_RATE - config.LEARNING_RATE_END) * progress_remaining


def stage_a_initial_ppo(args):
    """Stage A: Train initial PPO policy without obstacles."""
    print("=" * 60)
    print("[STAGE A] Initial PPO Training (straight cave, no obstacles)")
    print("=" * 60)

    env = make_env(headless=args.headless)

    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,  # Linear decay: 1e-4 → 1e-5 (MAVRL style)
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        policy_kwargs={
            "net_arch": config.ACTOR_HIDDEN,
            "activation_fn": torch.nn.ReLU,
        },
        tensorboard_log=str(config.LOG_DIR),
        verbose=1,
    )

    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    from stable_baselines3.common.callbacks import CheckpointCallback

    callbacks = [
        CheckpointCallback(
            save_freq=config.SAVE_FREQ,
            save_path=str(checkpoint_dir),
            name_prefix="stage_a",
        ),
    ]

    try:
        model.learn(
            total_timesteps=200_000,
            callback=callbacks,
            tb_log_name="stage_a_initial",
        )
    except KeyboardInterrupt:
        print("\n[STAGE A] Interrupted. Saving...")

    save_path = checkpoint_dir / "stage_a_final.zip"
    model.save(str(save_path))
    print(f"[STAGE A] Saved to {save_path}")
    env.close()
    return save_path


def stage_b_collect_data(args):
    """Stage B: Collect depth image sequences for VAE/LSTM training."""
    print("=" * 60)
    print("[STAGE B] Collecting Depth Data")
    print("=" * 60)

    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use reactive controller to collect data
    # Subscribe to depth map and save sequences
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    from cv_bridge import CvBridge

    rclpy.init()
    node = Node("data_collector")
    bridge = CvBridge()

    images = []
    states = []
    target_samples = 50_000

    def depth_cb(msg):
        try:
            depth = bridge.imgmsg_to_cv2(msg, 'passthrough')
            depth_m = depth / 1000.0 if depth.max() > 100 else depth
            depth_clamped = np.clip(depth_m, config.DEPTH_MIN, config.DEPTH_MAX)
            depth_norm = (depth_clamped / config.DEPTH_MAX * 255.0).astype(np.uint8)
            depth_resized = cv2.resize(depth_norm, (config.DEPTH_WIDTH, config.DEPTH_HEIGHT))
            images.append(depth_resized)
        except Exception:
            pass

    def odom_cb(msg):
        try:
            pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.x * q.x + q.z * q.z)
            yaw = np.arctan2(siny, cosy)
            state = pos + vel + [yaw]
            states.append(state)
        except Exception:
            pass

    node.create_subscription(Image, config.TOPIC_DEPTH_MAP, depth_cb, 10)
    node.create_subscription(Odometry, config.TOPIC_ODOM, odom_cb, 10)

    print(f"[STAGE B] Collecting {target_samples} samples...")
    print("[STAGE B] Make sure simulation is running!")

    start_time = time.time()
    while len(images) < target_samples and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
        if len(images) % 1000 == 0 and len(images) > 0:
            elapsed = time.time() - start_time
            print(f"[STAGE B] {len(images)}/{target_samples} ({elapsed:.0f}s)")

    # Save collected data
    if images and states:
        n = min(len(images), len(states))
        images_arr = np.array(images[:n])
        states_arr = np.array(states[:n])

        save_path = data_dir / "depth_sequences.npz"
        np.savez(save_path, images=images_arr, states=states_arr)
        print(f"[STAGE B] Saved {n} samples to {save_path}")

    node.destroy_node()
    rclpy.shutdown()
    return save_path if images else None


def stage_c_train_vae_lstm(args):
    """Stage C: Train VAE on depth data, then train LSTM."""
    print("=" * 60)
    print("[STAGE C] Training VAE + LSTM")
    print("=" * 60)

    data_path = Path(config.DATA_DIR) / "depth_sequences.npz"
    if not data_path.exists():
        print("[STAGE C] No data found. Run stage B first!")
        return None

    data = np.load(data_path)
    images = data['images']
    print(f"[STAGE C] Loaded {len(images)} depth images")

    # TODO: Implement VAE training
    # TODO: Implement LSTM training
    print("[STAGE C] VAE + LSTM training not yet implemented")
    print("[STAGE C] For now, using random encoder weights")

    return None


def stage_d_retrain_ppo(args):
    """Stage D: Retrain PPO with frozen encoder."""
    print("=" * 60)
    print("[STAGE D] Retraining PPO with frozen encoder")
    print("=" * 60)

    # Load stage A checkpoint or start fresh
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    stage_a_path = checkpoint_dir / "stage_a_final.zip"

    env = make_env(headless=args.headless)

    from stable_baselines3 import PPO

    if stage_a_path.exists() and not args.no_bc:
        print(f"[STAGE D] Loading from {stage_a_path}")
        model = PPO.load(str(stage_a_path), env=env)
    else:
        print("[STAGE D] Starting fresh PPO")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,  # Linear decay: 1e-4 → 1e-5 (MAVRL style)
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            gamma=config.GAMMA,
            gae_lambda=config.GAE_LAMBDA,
            clip_range=config.CLIP_RANGE,
            ent_coef=config.ENT_COEF,
            vf_coef=config.VF_COEF,
            max_grad_norm=config.MAX_GRAD_NORM,
            policy_kwargs={
                "net_arch": config.ACTOR_HIDDEN,
                "activation_fn": torch.nn.ReLU,
            },
            tensorboard_log=str(config.LOG_DIR),
            verbose=1,
        )

    callbacks = []

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name="stage_d_retrain",
        )
    except KeyboardInterrupt:
        print("\n[STAGE D] Interrupted. Saving...")

    save_path = checkpoint_dir / "final_model.zip"
    model.save(str(save_path))
    print(f"[STAGE D] Saved to {save_path}")
    env.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Drone MAVRL Training")
    parser.add_argument("--stage", type=str, default="d",
                       choices=["a", "b", "c", "d", "all"],
                       help="Training stage to run")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run Gazebo headless")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS,
                       help=f"Total timesteps (default: {config.TOTAL_TIMESTEPS})")
    parser.add_argument("--no-bc", action="store_true",
                       help="Skip BC warm-start, train from scratch")
    args = parser.parse_args()

    if args.stage == "a" or args.stage == "all":
        stage_a_initial_ppo(args)

    if args.stage == "b" or args.stage == "all":
        stage_b_collect_data(args)

    if args.stage == "c" or args.stage == "all":
        stage_c_train_vae_lstm(args)

    if args.stage == "d" or args.stage == "all":
        if args.resume:
            print(f"Resuming from {args.resume}")
            env = make_env(headless=args.headless)
            from stable_baselines3 import PPO
            model = PPO.load(args.resume, env=env)
            model.learn(total_timesteps=args.timesteps)
            model.save(str(Path(config.CHECKPOINT_DIR) / "final_model.zip"))
            env.close()
        else:
            stage_d_retrain_ppo(args)


if __name__ == "__main__":
    main()
