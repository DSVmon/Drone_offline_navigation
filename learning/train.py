#!/usr/bin/env python3
"""
Training script for MAVRL-style drone navigation.

Pipeline (matching MAVRL exactly):
    Stage A: Initial PPO with RecurrentPolicy (random encoder, no obstacles)
    Stage B: Collect depth data using initial policy
    Stage C: Train VAE on depth data
    Stage D: Retrain PPO with frozen VAE encoder + trained LSTM

Usage:
    python3 learning/train.py --stage a          # Stage A only
    python3 learning/train.py --stage b          # Stage B only
    python3 learning/train.py --stage c          # Stage C only
    python3 learning/train.py --stage d          # Stage D only
    python3 learning/train.py --stage all        # Full pipeline
    python3 learning/train.py --resume <path>    # Resume from checkpoint
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
from vec_drone_env import VecDroneEnv, EvalDroneEnv
from policy import RecurrentPolicy
from recurrent_ppo import RecurrentPPO


def make_env(headless=True, node_name="drone_env_node"):
    return VecDroneEnv(num_envs=1, headless=headless)


def make_eval_env(headless=True):
    return EvalDroneEnv(headless=headless)


def lr_schedule(progress_remaining):
    """Linear LR decay from LEARNING_RATE to LEARNING_RATE_END.
    Matching MAVRL: learning_rate_schedule(progress_remaining)."""
    return config.LEARNING_RATE_END + (config.LEARNING_RATE - config.LEARNING_RATE_END) * progress_remaining


def create_policy(device='cpu'):
    """Create RecurrentPolicy with MAVRL architecture."""
    policy = RecurrentPolicy(
        features_dim=config.FEATURES_DIM,
        lstm_hidden=config.LSTM_HIDDEN_SIZE,
        act_dim=config.ACTION_DIM,
        states_dim=config.STATE_DIM,
    )
    policy = policy.to(device)
    return policy


def stage_a_initial_ppo(args):
    """
    Stage A: Train initial PPO policy with random encoder.
    No obstacles, straight cave. MAVRL: ~200 iterations (200K steps).
    """
    print("=" * 60)
    print("[STAGE A] Initial PPO Training (random encoder)")
    print("=" * 60)

    env = make_env(headless=args.headless)
    eval_env = make_eval_env(headless=args.headless)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[STAGE A] Device: {device}")

    policy = create_policy(device)

    # Create RecurrentPPO (matching MAVRL hyperparameters)
    model = RecurrentPPO(
        policy=policy,
        env=env,
        lr=lr_schedule(1),  # Initial LR
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        device=device,
        tensorboard_log=str(config.LOG_DIR),
        lr_schedule=lr_schedule,
        eval_env=eval_env,
        eval_freq=config.EVAL_FREQ,
        eval_episodes=config.EVAL_EPISODES,
    )

    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Save checkpoint callback
    save_freq = config.SAVE_FREQ // config.N_STEPS  # Convert steps to iterations

    def save_callback(model):
        if model.num_updates % save_freq == 0 and model.num_updates > 0:
            path = checkpoint_dir / f"stage_a_iter_{model.num_updates:05d}.pth"
            model.save(str(path))

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=save_callback,
            log_interval=10,
        )
    except KeyboardInterrupt:
        print("\n[STAGE A] Interrupted. Saving...")

    save_path = checkpoint_dir / "stage_a_final.pth"
    model.save(str(save_path))
    print(f"[STAGE A] Saved to {save_path}")
    env.close()
    return save_path


def stage_b_collect_data(args):
    """
    Stage B: Collect depth image sequences using initial policy.
    MAVRL: collect_data.py - runs policy, saves depth + state sequences.
    """
    print("=" * 60)
    print("[STAGE B] Collecting Depth Data with Initial Policy")
    print("=" * 60)

    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load initial policy from Stage A
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    stage_a_path = checkpoint_dir / "stage_a_final.pth"

    if not stage_a_path.exists():
        print(f"[STAGE B] No Stage A checkpoint found at {stage_a_path}")
        print("[STAGE B] Run Stage A first!")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = create_policy(device)

    checkpoint = torch.load(stage_a_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    print(f"[STAGE B] Loaded policy from {stage_a_path}")

    # Collect data
    env = make_env(headless=args.headless)
    target_sequences = 500  # Number of trajectory sequences
    seq_length = 100        # Steps per sequence

    all_images = []
    all_states = []
    all_lstm_h = []
    all_lstm_c = []

    for seq_idx in range(target_sequences):
        obs = env.reset()
        lstm_h, lstm_c = policy.get_initial_hidden(1, device)

        seq_images = []
        seq_states = []
        seq_lstm_h = []
        seq_lstm_c = []

        for step in range(seq_length):
            image = torch.FloatTensor(obs['image']).unsqueeze(0).to(device)
            state = torch.FloatTensor(obs['state']).unsqueeze(0).to(device)

            with torch.no_grad():
                # Run encoder + LSTM
                latent_pi, latent_vf, new_h, new_c = policy.forward_rnn(
                    image, state, (lstm_h, lstm_c)
                )

            # Store raw depth + state (for VAE training)
            seq_images.append(obs['image'].copy())
            seq_states.append(obs['state'].copy())
            seq_lstm_h.append(lstm_h.cpu().numpy())
            seq_lstm_c.append(lstm_c.cpu().numpy())

            # Take action (deterministic)
            action_mean, _ = policy.forward_from_latent(latent_pi, latent_vf)
            action = torch.tanh(action_mean).cpu().numpy()[0]

            obs, reward, done, info = env.step(action)
            lstm_h, lstm_c = new_h.detach(), new_c.detach()

            if done:
                break

        all_images.extend(seq_images)
        all_states.extend(seq_states)
        all_lstm_h.extend(seq_lstm_h)
        all_lstm_c.extend(seq_lstm_c)

        if (seq_idx + 1) % 50 == 0:
            print(f"[STAGE B] {seq_idx + 1}/{target_sequences} sequences collected")

    # Save collected data
    if all_images:
        save_path = data_dir / "lstm_dataset"
        save_path.mkdir(parents=True, exist_ok=True)

        images_arr = np.array(all_images, dtype=np.uint8)
        states_arr = np.array(all_states, dtype=np.float32)
        lstm_h_arr = np.array([h[0][0] for h in all_lstm_h], dtype=np.float32)
        lstm_c_arr = np.array([c[0][0] for c in all_lstm_c], dtype=np.float32)

        np.savez(save_path / "data.npz",
                 images=images_arr,
                 states=states_arr,
                 lstm_h=lstm_h_arr,
                 lstm_c=lstm_c_arr)
        print(f"[STAGE B] Saved {len(all_images)} samples to {save_path}")

    env.close()
    return save_path if all_images else None


def stage_c_train_vae(args):
    """
    Stage C: Train VAE on collected depth data.
    MAVRL: trainvae.py - trains VAE encoder for depth reconstruction.
    Saves to: data/vae/best.tar
    """
    print("=" * 60)
    print("[STAGE C] Training VAE")
    print("=" * 60)

    data_dir = Path(config.DATA_DIR) / "lstm_dataset"
    if not (data_dir / "data.npz").exists():
        print(f"[STAGE C] No data found at {data_dir}/data.npz")
        print("[STAGE C] Run Stage B first!")
        return None

    from vae import train_vae

    save_path = Path(config.DATA_DIR) / "vae" / "best.tar"

    print(f"[STAGE C] Training VAE on {data_dir}")
    vae = train_vae(
        data_path=data_dir,
        save_path=save_path,
        epochs=args.vae_epochs,
        batch_size=32,
        lr=1e-3,
        patience=50,
    )

    print(f"[STAGE C] VAE saved to {save_path}")
    return save_path


def stage_c_plus_train_lstm(args):
    """
    Stage C+: Train LSTM on depth reconstruction (offline, no env).
    MAVRL: train_lstm_without_env.py
    - Loads VAE weights (encoder frozen)
    - Trains LSTM to predict future depth from sequence of latent vectors
    - Saves LSTM weights for Stage D
    """
    print("=" * 60)
    print("[STAGE C+] Training LSTM on Depth Reconstruction")
    print("=" * 60)

    from train_lstm import train_lstm
    return train_lstm(args)


def stage_d_retrain_ppo(args):
    """
    Stage D: Retrain PPO with frozen VAE encoder + trained LSTM.
    MAVRL: load VAE encoder weights -> freeze -> load LSTM weights -> freeze -> retrain PPO head.
    """
    print("=" * 60)
    print("[STAGE D] Retraining PPO with Frozen Encoder + LSTM")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create policy
    policy = create_policy(device)

    # Load VAE encoder weights (freeze encoder)
    vae_path = Path(config.DATA_DIR) / "vae" / "best.tar"
    if vae_path.exists():
        print(f"[STAGE D] Loading VAE encoder from {vae_path}")
        from vae import load_vae_encoder
        load_vae_encoder(vae_path, policy, device)

        # Freeze encoder
        for param in policy.encoder.parameters():
            param.requires_grad = False
        print("[STAGE D] Encoder frozen")
    else:
        print("[STAGE D] No VAE found, using random encoder weights")

    # Load trained LSTM weights (freeze LSTM) — Stage C+
    lstm_path = Path(config.DATA_DIR) / "lstm" / "best_lstm.tar"
    if lstm_path.exists():
        print(f"[STAGE D] Loading trained LSTM from {lstm_path}")
        from train_lstm import load_trained_lstm
        load_trained_lstm(policy, lstm_path, device)

        # Freeze LSTM
        for param in policy.lstm.parameters():
            param.requires_grad = False
        print("[STAGE D] LSTM frozen")
    else:
        print("[STAGE D] No trained LSTM found (Stage C+ not run)")

    # Load Stage A checkpoint if exists
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    stage_a_path = checkpoint_dir / "stage_a_final.pth"
    if stage_a_path.exists() and not args.no_bc:
        print(f"[STAGE D] Loading Stage A weights (encoder/LSTM will be overwritten)")
        stage_a_checkpoint = torch.load(stage_a_path, map_location=device)
        # Load non-encoder, non-LSTM weights from Stage A
        policy_state = policy.state_dict()
        loaded = 0
        for key in stage_a_checkpoint['policy_state_dict']:
            if not key.startswith('encoder.') and not key.startswith('lstm.'):
                if key in policy_state:
                    policy_state[key] = stage_a_checkpoint['policy_state_dict'][key]
                    loaded += 1
        policy.load_state_dict(policy_state)
        print(f"[STAGE D] Loaded {loaded} Stage A weights (actor/critic)")

    # Create environment
    env = make_env(headless=args.headless)
    eval_env = make_eval_env(headless=args.headless)

    # Create RecurrentPPO
    model = RecurrentPPO(
        policy=policy,
        env=env,
        lr=lr_schedule(1),
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        device=device,
        tensorboard_log=str(config.LOG_DIR),
        lr_schedule=lr_schedule,
        eval_env=eval_env,
        eval_freq=config.EVAL_FREQ,
        eval_episodes=config.EVAL_EPISODES,
    )

    # Load optimizer state from Stage A if available
    if stage_a_path.exists() and not args.no_bc:
        if 'optimizer_state_dict' in stage_a_checkpoint:
            try:
                model.optimizer.load_state_dict(stage_a_checkpoint['optimizer_state_dict'])
                print("[STAGE D] Loaded optimizer state from Stage A")
            except Exception as e:
                print(f"[STAGE D] Could not load optimizer state: {e}")

    save_freq = config.SAVE_FREQ // config.N_STEPS

    def save_callback(model):
        if model.num_updates % save_freq == 0 and model.num_updates > 0:
            path = checkpoint_dir / f"stage_d_iter_{model.num_updates:05d}.pth"
            model.save(str(path))

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=save_callback,
            log_interval=10,
        )
    except KeyboardInterrupt:
        print("\n[STAGE D] Interrupted. Saving...")

    save_path = checkpoint_dir / "final_model.pth"
    model.save(str(save_path))
    print(f"[STAGE D] Saved to {save_path}")
    env.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Drone MAVRL Training")
    parser.add_argument("--stage", type=str, default="a",
                       choices=["a", "b", "c", "c+", "d", "all"],
                       help="Training stage to run")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run Gazebo headless")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS,
                       help=f"Total timesteps (default: {config.TOTAL_TIMESTEPS})")
    parser.add_argument("--no-bc", action="store_true",
                       help="Don't load Stage A weights, train from scratch")
    parser.add_argument("--vae-epochs", type=int, default=1000,
                       help="VAE training epochs (Stage C, matching MAVRL)")
    parser.add_argument("--lstm-epochs", type=int, default=2000,
                       help="LSTM training epochs (Stage C+, matching MAVRL)")
    parser.add_argument("--lstm-seq-len", type=int, default=10,
                       help="LSTM sequence length for Stage C+")
    parser.add_argument("--recon", nargs='+', type=int, default=[0, 0, 1],
                       help="Reconstruct [past, current, future] for Stage C+")
    args = parser.parse_args()

    args.recon = [bool(x) for x in args.recon]

    print(f"[TRAIN] Stage: {args.stage}")
    print(f"[TRAIN] Timesteps: {args.timesteps:,}")
    print(f"[TRAIN] Headless: {args.headless}")

    if args.resume:
        print(f"[TRAIN] Resuming from {args.resume}")
        # TODO: implement resume logic
        return

    if args.stage == "a" or args.stage == "all":
        stage_a_initial_ppo(args)

    if args.stage == "b" or args.stage == "all":
        stage_b_collect_data(args)

    if args.stage == "c" or args.stage == "all":
        stage_c_train_vae(args)

    if args.stage == "c+" or args.stage == "all":
        stage_c_plus_train_lstm(args)

    if args.stage == "d" or args.stage == "all":
        stage_d_retrain_ppo(args)


if __name__ == "__main__":
    main()
