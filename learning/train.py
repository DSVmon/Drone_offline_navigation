#!/usr/bin/env python3
"""
Main training script: Behavior Cloning → PPO fine-tuning.

Usage:
    python3 learning/train.py                       # BC → PPO fresh
    python3 learning/train.py --resume <checkpoint>  # Resume PPO from checkpoint
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

import config
from drone_env import DroneEnv
from bc_model import train_bc, load_bc_into_ppo
from callbacks import ConsoleMonitorCallback, CSVEpisodeLogger


def main():
    parser = argparse.ArgumentParser(description="Drone NN Control Training")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to PPO checkpoint to resume from",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run Gazebo headless",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=config.TOTAL_TIMESTEPS,
        help=f"Total PPO timesteps (default: {config.TOTAL_TIMESTEPS})",
    )
    parser.add_argument(
        "--no-bc",
        action="store_true",
        help="Skip BC warm-start, train PPO from scratch",
    )
    args = parser.parse_args()

    # --- Phase 1: BC warm-start (unless resuming or --no-bc) ---
    if args.resume is None and not args.no_bc:
        expert_dir = Path(config.EXPERT_DIR)
        npz_files = list(expert_dir.glob("*.npz"))

        if not npz_files:
            print("[TRAIN] No expert data found. Run collect_expert.py first!")
            print("[TRAIN] Example:")
            print("  Terminal 1: ./run_drone.sh")
            print("  Terminal 2: python3 learning/collect_expert.py")
            print("[TRAIN] After collection, re-run this script.")
            sys.exit(1)

        print("=" * 60)
        print("[TRAIN] Phase 1: Behavior Cloning (BC warm-start)")
        print("=" * 60)
        bc_model = train_bc()

        print("=" * 60)
        print("[TRAIN] Creating PPO model with BC warm-start...")
        print("=" * 60)
        env = DroneEnv(headless=args.headless)

        from stable_baselines3 import PPO

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
            verbose=1,
        )

        load_bc_into_ppo(bc_model, model.policy)
        print("[TRAIN] BC warm-start complete. Starting PPO fine-tuning.")

    elif args.resume is not None:
        # --- Resume from checkpoint ---
        print("=" * 60)
        print(f"[TRAIN] Resuming from checkpoint: {args.resume}")
        print("=" * 60)
        env = DroneEnv(headless=args.headless)

        from stable_baselines3 import PPO

        model = PPO.load(args.resume, env=env)
        print(f"[TRAIN] Loaded checkpoint at step {model.num_timesteps}")

    else:
        # --- PPO from scratch (--no-bc) ---
        print("=" * 60)
        print("[TRAIN] PPO from scratch (no BC warm-start)")
        print("=" * 60)
        env = DroneEnv(headless=args.headless)

        from stable_baselines3 import PPO

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
            verbose=1,
        )

    # --- Phase 2: PPO training ---
    print("=" * 60)
    print(f"[TRAIN] Phase 2: PPO fine-tuning for {args.timesteps:_} steps")
    print(f"        Headless: {args.headless}")
    print(f"        Cave change interval: {config.CAVE_CHANGE_INTERVAL} episodes")
    print(f"        Checkpoints: {config.CHECKPOINT_DIR}")
    print(f"        TensorBoard: {config.LOG_DIR}")
    print("=" * 60)

    from stable_baselines3.common.callbacks import CheckpointCallback

    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=config.SAVE_FREQ,
            save_path=str(checkpoint_dir),
            name_prefix="rl_model",
        ),
        ConsoleMonitorCallback(),
        CSVEpisodeLogger(),
    ]

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=(args.resume is None),
            tb_log_name="ppo_drone",
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user. Saving checkpoint...")

    # --- Save final model ---
    final_path = checkpoint_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"[TRAIN] Final model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
