#!/usr/bin/env python3
"""
Training script for MAVRL-style drone navigation.

Pipeline (matching MAVRL exactly):
    Stage A: Initial PPO with RecurrentPolicy (random encoder, no obstacles)
    Stage B: Collect depth data using initial policy
    Stage C: Train VAE on depth data
    Stage C+: Train LSTM on depth reconstruction
    Stage D: Retrain PPO with frozen VAE encoder + trained LSTM

Resume: all stages support --resume to continue from last checkpoint.

Usage:
    python3 learning/train.py --stage a          # Stage A only
    python3 learning/train.py --stage all        # Full pipeline
    python3 learning/train.py --stage a --resume # Resume Stage A
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
    """Linear LR decay from LEARNING_RATE to LEARNING_RATE_END."""
    return config.LEARNING_RATE_END + (config.LEARNING_RATE - config.LEARNING_RATE_END) * progress_remaining


def create_policy(device='cpu'):
    """Create RecurrentPolicy with MAVRL architecture."""
    policy = RecurrentPolicy(
        features_dim=config.FEATURES_DIM,
        lstm_hidden=config.LSTM_HIDDEN_SIZE,
        act_dim=config.ACTION_DIM,
        states_dim=config.STATE_DIM,
    )
    return policy.to(device)


def find_last_checkpoint(checkpoint_dir, prefix):
    """Find the last numbered checkpoint for a stage."""
    checkpoints = sorted(checkpoint_dir.glob(f"{prefix}_iter_*.pth"))
    if checkpoints:
        return checkpoints[-1]
    return None


def resume_training(model, checkpoint_path, device):
    """Resume training from a checkpoint."""
    print(f"[RESUME] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.policy.load_state_dict(checkpoint['policy_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.num_timesteps = checkpoint['num_timesteps']
    model.num_updates = checkpoint['num_updates']
    print(f"[RESUME] Resumed at step {model.num_timesteps:,}, update {model.num_updates}")
    return model


# ═══════════════════════════════════════════════════════════════
# STAGE A: Initial PPO
# ═══════════════════════════════════════════════════════════════

def stage_a_initial_ppo(args):
    """
    Stage A: Train initial PPO policy.
    Supports resume from last checkpoint.
    """
    print("=" * 60)
    print("[STAGE A] Initial PPO Training")
    print("=" * 60)

    env = make_env(headless=args.headless)
    eval_env = make_eval_env(headless=args.headless)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[STAGE A] Device: {device}")

    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    policy = create_policy(device)

    # Check for resume
    resume_path = None
    if args.resume:
        # Try to find last checkpoint
        last_ckpt = find_last_checkpoint(checkpoint_dir, "stage_a")
        if last_ckpt:
            resume_path = last_ckpt
            print(f"[STAGE A] Found checkpoint: {resume_path}")
        else:
            print("[STAGE A] No checkpoint found, starting fresh")
    elif not args.no_bc:
        # Try MAVRL weights as warm start
        from vae import try_load_mavrl_encoder, try_load_mavrl_lstm
        try_load_mavrl_encoder(policy, device)
        try_load_mavrl_lstm(policy, device)

    # Create RecurrentPPO
    model = RecurrentPPO(
        policy=policy, env=env, lr=lr_schedule(1),
        gamma=config.GAMMA, gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE, ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF, max_grad_norm=config.MAX_GRAD_NORM,
        n_steps=config.N_STEPS, batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS, device=device,
        tensorboard_log=str(config.LOG_DIR), lr_schedule=lr_schedule,
        eval_env=eval_env, eval_freq=config.EVAL_FREQ,
        eval_episodes=config.EVAL_EPISODES,
    )

    # Resume if checkpoint found
    if resume_path:
        model = resume_training(model, resume_path, device)

    save_freq = config.SAVE_FREQ // config.N_STEPS

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


# ═══════════════════════════════════════════════════════════════
# STAGE B: Collect Data
# ═══════════════════════════════════════════════════════════════

def stage_b_collect_data(args):
    """
    Stage B: Collect depth image sequences.
    Supports resume by appending to existing data.
    """
    print("=" * 60)
    print("[STAGE B] Collecting Depth Data")
    print("=" * 60)

    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / "lstm_dataset"
    save_path.mkdir(parents=True, exist_ok=True)
    data_file = save_path / "data.npz"

    # Check existing data for resume
    existing_samples = 0
    if args.resume and data_file.exists():
        existing_data = np.load(data_file)
        existing_samples = len(existing_data['images'])
        print(f"[STAGE B] Found {existing_samples} existing samples")

    # Load policy
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    stage_a_path = checkpoint_dir / "stage_a_final.pth"

    if not stage_a_path.exists():
        print(f"[STAGE B] No Stage A checkpoint. Run Stage A first!")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = create_policy(device)
    checkpoint = torch.load(stage_a_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    # Calculate remaining sequences
    target_total = args.num_sequences * args.seq_length
    target_sequences = args.num_sequences

    if existing_samples > 0:
        collected_sequences = existing_samples // args.seq_length
        remaining = target_sequences - collected_sequences
        if remaining <= 0:
            print(f"[STAGE B] Already have {existing_samples} samples (target: {target_total})")
            print("[STAGE B] Skipping data collection")
            return save_path
        print(f"[STAGE B] Resuming: {existing_samples} samples, collecting {remaining} more sequences")
        target_sequences = remaining

    # Collect data
    env = make_env(headless=args.headless)
    all_images = []
    all_states = []

    for seq_idx in range(target_sequences):
        obs, _ = env.reset()
        lstm_h, lstm_c = policy.get_initial_hidden(1, device)

        for step in range(args.seq_length):
            img = torch.FloatTensor(obs['image']).reshape(1, 1, 1, 256, 256).to(device)
            if img.max() > 1.0:
                img = img.float() / 255.0
            st = torch.FloatTensor(obs['state']).reshape(1, 1, 7).to(device)

            with torch.no_grad():
                lp, lv, nh, nc = policy.forward_rnn(img, st, (lstm_h, lstm_c))
                am, _ = policy.forward_from_latent(lp, lv)
                action = torch.tanh(am).cpu().numpy()[0]

            all_images.append(obs['image'].copy())
            all_states.append(obs['state'].copy())
            obs, _, done, _ = env.step(action)
            lstm_h, lstm_c = nh.detach(), nc.detach()
            if done:
                break

        if (seq_idx + 1) % 50 == 0:
            print(f"[STAGE B] {seq_idx + 1}/{target_sequences} sequences "
                  f"({existing_samples + len(all_images)} total)")

    # Merge with existing data
    if existing_samples > 0:
        existing_images = np.load(data_file)['images']
        existing_states = np.load(data_file)['states']
        all_images = list(existing_images) + all_images
        all_states = list(existing_states) + all_states

    # Save
    np.savez(data_file,
             images=np.array(all_images, dtype=np.uint8),
             states=np.array(all_states, dtype=np.float32))
    print(f"[STAGE B] Total saved: {len(all_images)} samples")
    env.close()
    return save_path


# ═══════════════════════════════════════════════════════════════
# STAGE C: Train VAE
# ═══════════════════════════════════════════════════════════════

def stage_c_train_vae(args):
    """
    Stage C: Train VAE.
    Supports resume via train_vae() which checks for existing best.tar.
    """
    print("=" * 60)
    print("[STAGE C] Training VAE")
    print("=" * 60)

    data_dir = Path(config.DATA_DIR) / "lstm_dataset"
    if not (data_dir / "data.npz").exists():
        print(f"[STAGE C] No data found. Run Stage B first!")
        return None

    from vae import train_vae
    save_path = Path(config.DATA_DIR) / "vae" / "best.tar"

    # Resume: train_vae checks for existing best.tar
    if args.resume and save_path.exists():
        print(f"[STAGE C] Found existing VAE at {save_path}")
        print("[STAGE C] Will resume training from there")

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


# ═══════════════════════════════════════════════════════════════
# STAGE C+: Train LSTM
# ═══════════════════════════════════════════════════════════════

def stage_c_plus_train_lstm(args):
    """
    Stage C+: Train LSTM on depth reconstruction.
    Supports resume via train_lstm() which checks for existing best_lstm.tar.
    """
    print("=" * 60)
    print("[STAGE C+] Training LSTM on Depth Reconstruction")
    print("=" * 60)

    lstm_path = Path(config.DATA_DIR) / "lstm" / "best_lstm.tar"
    if args.resume and lstm_path.exists():
        print(f"[STAGE C+] Found existing LSTM at {lstm_path}")
        print("[STAGE C+] Will resume training from there")

    from train_lstm import train_lstm
    return train_lstm(args)


# ═══════════════════════════════════════════════════════════════
# STAGE D: Retrain PPO
# ═══════════════════════════════════════════════════════════════

def stage_d_retrain_ppo(args):
    """
    Stage D: Retrain PPO with frozen encoder + LSTM.
    Supports resume from last checkpoint.
    """
    print("=" * 60)
    print("[STAGE D] Retraining PPO with Frozen Encoder + LSTM")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create policy
    policy = create_policy(device)

    # Load VAE encoder (freeze)
    vae_path = Path(config.DATA_DIR) / "vae" / "best.tar"
    if vae_path.exists():
        from vae import load_vae_encoder
        load_vae_encoder(vae_path, policy, device)
        for param in policy.encoder.parameters():
            param.requires_grad = False
        print("[STAGE D] Encoder frozen")
    else:
        print("[STAGE D] No VAE found, using random encoder")

    # Load trained LSTM (freeze)
    lstm_path = Path(config.DATA_DIR) / "lstm" / "best_lstm.tar"
    if lstm_path.exists():
        from train_lstm import load_trained_lstm
        load_trained_lstm(policy, lstm_path, device)
        for param in policy.lstm.parameters():
            param.requires_grad = False
        print("[STAGE D] LSTM frozen")
    else:
        print("[STAGE D] No trained LSTM found")

    # Load Stage A MLP weights
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    stage_a_path = checkpoint_dir / "stage_a_final.pth"
    if stage_a_path.exists() and not args.no_bc:
        stage_a_checkpoint = torch.load(stage_a_path, map_location=device)
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
        policy=policy, env=env, lr=lr_schedule(1),
        gamma=config.GAMMA, gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE, ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF, max_grad_norm=config.MAX_GRAD_NORM,
        n_steps=config.N_STEPS, batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS, device=device,
        tensorboard_log=str(config.LOG_DIR), lr_schedule=lr_schedule,
        eval_env=eval_env, eval_freq=config.EVAL_FREQ,
        eval_episodes=config.EVAL_EPISODES,
    )

    # Check for resume
    resume_path = None
    if args.resume:
        last_ckpt = find_last_checkpoint(checkpoint_dir, "stage_d")
        if last_ckpt:
            resume_path = last_ckpt
            print(f"[STAGE D] Found checkpoint: {resume_path}")

    if resume_path:
        model = resume_training(model, resume_path, device)

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


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Drone MAVRL Training")
    parser.add_argument("--stage", type=str, default="a",
                       choices=["a", "b", "c", "c+", "d", "all"],
                       help="Training stage to run")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run Gazebo headless")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS,
                       help=f"Total timesteps (default: {config.TOTAL_TIMESTEPS})")
    parser.add_argument("--no-bc", action="store_true",
                       help="Don't load Stage A weights, train from scratch")
    parser.add_argument("--vae-epochs", type=int, default=1000,
                       help="VAE training epochs (Stage C)")
    parser.add_argument("--lstm-epochs", type=int, default=2000,
                       help="LSTM training epochs (Stage C+)")
    parser.add_argument("--lstm-seq-len", type=int, default=10,
                       help="LSTM sequence length for Stage C+")
    parser.add_argument("--recon", nargs='+', type=int, default=[0, 0, 1],
                       help="Reconstruct [past, current, future] for Stage C+")
    parser.add_argument("--num-sequences", type=int, default=1000,
                       help="Number of trajectory sequences for Stage B")
    parser.add_argument("--seq-length", type=int, default=1000,
                       help="Steps per sequence for Stage B")
    args = parser.parse_args()

    args.recon = [bool(x) for x in args.recon]

    print(f"[TRAIN] Stage: {args.stage}")
    print(f"[TRAIN] Resume: {args.resume}")
    print(f"[TRAIN] Timesteps: {args.timesteps:,}")

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
