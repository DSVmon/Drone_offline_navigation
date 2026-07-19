#!/usr/bin/env python3
"""
Collect depth data for VAE training using trained policy.

MAVRL: collect_data.py - runs policy in env, saves depth sequences.
Output: data/lstm_dataset/data.npz

Usage:
    python3 learning/collect_data.py --checkpoint learning/checkpoints/stage_a_final.pth
    python3 learning/collect_data.py --sequences 500
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config
from drone_env import DroneEnv
from policy import RecurrentPolicy


def collect_data(checkpoint_path, num_sequences=500, seq_length=100,
                 headless=True, output_dir=None):
    """
    Collect depth image sequences using trained policy.

    MAVRL flow:
        1. Load trained policy from Stage A
        2. Run policy in environment
        3. Save depth images + states + LSTM states
    """
    print("=" * 60)
    print("[COLLECT] Collecting Depth Data")
    print("=" * 60)

    # Setup paths
    if output_dir is None:
        output_dir = Path(config.DATA_DIR) / "lstm_dataset"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[COLLECT] Device: {device}")

    # Create and load policy
    policy = RecurrentPolicy(
        features_dim=config.FEATURES_DIM,
        lstm_hidden=config.LSTM_HIDDEN_SIZE,
        act_dim=config.ACTION_DIM,
        states_dim=config.STATE_DIM,
    ).to(device)

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"[COLLECT] Loaded policy from {checkpoint_path}")
    else:
        print(f"[COLLECT] WARNING: No checkpoint at {checkpoint_path}, using random weights")

    policy.eval()

    # Create environment
    env = DroneEnv(headless=headless, node_name="data_collector")

    # Collect data
    all_images = []
    all_states = []
    all_lstm_h = []
    all_lstm_c = []

    print(f"[COLLECT] Collecting {num_sequences} sequences of {seq_length} steps...")
    start_time = time.time()

    for seq_idx in range(num_sequences):
        obs, _ = env.reset()
        lstm_h, lstm_c = policy.get_initial_hidden(1, device)

        for step in range(seq_length):
            # Prepare observation
            image = torch.FloatTensor(obs['image']).unsqueeze(0).unsqueeze(0).to(device)
            state = torch.FloatTensor(obs['state']).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                # Run encoder + LSTM
                latent_pi, latent_vf, new_h, new_c = policy.forward_rnn(
                    image, state, (lstm_h, lstm_c)
                )

                # Get action
                action_mean, _ = policy.forward_from_latent(latent_pi, latent_vf)
                action = torch.tanh(action_mean).cpu().numpy()[0]

            # Store raw data
            all_images.append(obs['image'].copy())
            all_states.append(obs['state'].copy())
            all_lstm_h.append(lstm_h.cpu().numpy()[0][0])
            all_lstm_c.append(lstm_c.cpu().numpy()[0][0])

            # Step environment
            obs, reward, done, info = env.step(action)
            lstm_h, lstm_c = new_h.detach(), new_c.detach()

            if done:
                break

        # Progress
        if (seq_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (seq_idx + 1) / elapsed
            eta = (num_sequences - seq_idx - 1) / rate
            print(f"[COLLECT] {seq_idx + 1}/{num_sequences} sequences "
                  f"({elapsed:.0f}s, ETA: {eta:.0f}s)")

    # Save data
    images_arr = np.array(all_images, dtype=np.uint8)
    states_arr = np.array(all_states, dtype=np.float32)
    lstm_h_arr = np.array(all_lstm_h, dtype=np.float32)
    lstm_c_arr = np.array(all_lstm_c, dtype=np.float32)

    save_path = output_dir / "data.npz"
    np.savez(save_path,
             images=images_arr,
             states=states_arr,
             lstm_h=lstm_h_arr,
             lstm_c=lstm_c_arr)

    elapsed = time.time() - start_time
    print(f"[COLLECT] Saved {len(all_images)} samples to {save_path}")
    print(f"[COLLECT] Images: {images_arr.shape}, States: {states_arr.shape}")
    print(f"[COLLECT] Total time: {elapsed:.0f}s")

    env.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Collect depth data for VAE training")
    parser.add_argument("--checkpoint", type=str,
                        default=str(config.CHECKPOINT_DIR / "stage_a_final.pth"),
                        help="Path to trained policy checkpoint")
    parser.add_argument("--sequences", type=int, default=500,
                        help="Number of trajectory sequences to collect")
    parser.add_argument("--seq-length", type=int, default=100,
                        help="Steps per sequence")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run Gazebo headless")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/lstm_dataset)")
    args = parser.parse_args()

    collect_data(
        checkpoint_path=args.checkpoint,
        num_sequences=args.sequences,
        seq_length=args.seq_length,
        headless=args.headless,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
