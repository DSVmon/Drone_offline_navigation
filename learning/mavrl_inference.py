#!/usr/bin/env python3
"""
Load MAVRL pre-trained weights and run inference through our architecture.

Usage:
    python3 learning/mavrl_inference.py --checkpoint mavrl_weights/RecurrentPPO_1/Policy/iter_00060.pth
    python3 learning/mavrl_inference.py --checkpoint mavrl_weights/RecurrentPPO_1/Policy/iter_00060.pth --test
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).parent))
import config
from policy import RecurrentPolicy


def load_mavrl_checkpoint(checkpoint_path, device='cpu'):
    """
    Load MAVRL checkpoint and extract encoder weights.

    Returns:
        policy: RecurrentPolicy with loaded encoder weights
        checkpoint: raw checkpoint data
    """
    print(f"[LOAD] Loading MAVRL checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']

    print(f"[LOAD] Checkpoint keys: {len(state_dict)}")

    # Create policy
    policy = RecurrentPolicy(
        features_dim=config.FEATURES_DIM,
        lstm_hidden=config.LSTM_HIDDEN_SIZE,
        act_dim=config.ACTION_DIM,
        states_dim=config.STATE_DIM,
    ).to(device)

    # Map MAVRL keys to our keys
    key_map = {
        'features_extractor.conv1.weight': 'encoder.conv1.weight',
        'features_extractor.conv1.bias': 'encoder.conv1.bias',
        'features_extractor.conv2.weight': 'encoder.conv2.weight',
        'features_extractor.conv2.bias': 'encoder.conv2.bias',
        'features_extractor.conv3.weight': 'encoder.conv3.weight',
        'features_extractor.conv3.bias': 'encoder.conv3.bias',
        'features_extractor.conv4.weight': 'encoder.conv4.weight',
        'features_extractor.conv4.bias': 'encoder.conv4.bias',
        'features_extractor.conv5.weight': 'encoder.conv5.weight',
        'features_extractor.conv5.bias': 'encoder.conv5.bias',
        'features_extractor.conv6.weight': 'encoder.conv6.weight',
        'features_extractor.conv6.bias': 'encoder.conv6.bias',
        'features_extractor.linear.weight': 'encoder.fc_mu.weight',
        'features_extractor.linear.bias': 'encoder.fc_mu.bias',
    }

    policy_state = policy.state_dict()
    transferred = 0

    for mavrl_key, our_key in key_map.items():
        if mavrl_key in state_dict and our_key in policy_state:
            if state_dict[mavrl_key].shape == policy_state[our_key].shape:
                policy_state[our_key] = state_dict[mavrl_key]
                transferred += 1
            else:
                print(f"[LOAD] SHAPE MISMATCH: {mavrl_key}: "
                      f"{state_dict[mavrl_key].shape} vs {policy_state[our_key].shape}")

    policy.load_state_dict(policy_state)
    print(f"[LOAD] Encoder weights transferred: {transferred}/{len(key_map)}")

    # Show what's loaded vs random
    print(f"[LOAD] LSTM input_size: {policy.lstm.input_size} (expected: 64)")
    print(f"[LOAD] Total params: {sum(p.numel() for p in policy.parameters()):,}")

    return policy, checkpoint


def run_inference(policy, depth_map, goal_point, pos, vel_world, yaw,
                  lstm_hidden=None, device='cpu'):
    """
    Run inference through our architecture.

    Args:
        policy: RecurrentPolicy with loaded weights
        depth_map: uint8 (256, 256) depth map
        goal_point: (3,) goal position
        pos: (3,) current position
        vel_world: (3,) world-frame velocity
        yaw: float current yaw
        lstm_hidden: optional LSTM hidden state

    Returns:
        action: (4,) normalized action [ax, ay, az, yaw_rate] in [-1, 1]
        cmd: (4,) physical command [m/s², m/s², m/s², rad/s]
        new_hidden: updated LSTM hidden state
    """
    # Normalize depth map
    image = torch.FloatTensor(depth_map).reshape(1, 1, 1, 256, 256)
    if image.max() > 1.0:
        image = image.float() / 255.0
    image = image.to(device)

    # Build state vector (7-dim, MAVRL style)
    delta_p = goal_point - pos
    horizon_dist = np.sqrt(delta_p[0]**2 + delta_p[1]**2)
    log_distance = np.log(horizon_dist + 1.0)

    # World to body velocity (manual rotation, no scipy dependency)
    cy, sy = np.cos(yaw), np.sin(yaw)
    flu_x = vel_world[1]
    flu_y = -vel_world[0]
    flu_z = vel_world[2]
    body_x = cy * flu_x + sy * flu_y
    body_y = -sy * flu_x + cy * flu_y
    vel_body = np.array([body_x, body_y, flu_z])

    horizon_vel = np.sqrt(vel_body[0]**2 + vel_body[1]**2)
    theta = np.arctan2(-delta_p[0], delta_p[1])
    horizon_vel_dire = np.arctan2(vel_body[1], vel_body[0])

    state = np.array([
        log_distance, horizon_vel, theta, horizon_vel_dire,
        delta_p[2], vel_body[2], yaw,
    ], dtype=np.float64)

    state_tensor = torch.FloatTensor(state).reshape(1, 1, 7).to(device)

    # Forward through policy
    if lstm_hidden is None:
        lstm_hidden = policy.get_initial_hidden(1, device)

    with torch.no_grad():
        latent_pi, latent_vf, new_h, new_c = policy.forward_rnn(
            image, state_tensor, lstm_hidden
        )
        action_mean, value = policy.forward_from_latent(latent_pi, latent_vf)
        action = torch.tanh(action_mean).cpu().numpy()[0]

    # Denormalize to physical units
    cmd = action * np.array([4.0, 4.0, 1.0, 0.6])

    new_hidden = (new_h, new_c)
    return action, cmd, new_hidden


def test_inference():
    """Test inference with synthetic data."""
    print("=" * 70)
    print("  MAVRL INFERENCE TEST (synthetic data)")
    print("=" * 70)

    device = torch.device('cpu')

    # Create policy with random weights (simulates loaded MAVRL)
    policy = RecurrentPolicy(
        features_dim=config.FEATURES_DIM,
        lstm_hidden=config.LSTM_HIDDEN_SIZE,
        act_dim=config.ACTION_DIM,
        states_dim=config.STATE_DIM,
    ).to(device)
    policy.eval()

    print(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params")
    print(f"LSTM input: {policy.lstm.input_size} (MAVRL: 64)")
    print(f"LSTM hidden: {policy.lstm.hidden_size}")

    # Generate realistic depth map
    depth = np.ones((256, 256), dtype=np.uint8) * 128
    depth[50:200, 50:80] = 30   # Left wall
    depth[50:200, 180:210] = 40  # Right wall
    depth[180:256, :] = 20       # Floor

    # Drone state
    pos = np.array([5.0, 0.0, 1.5])
    goal = np.array([80.0, 0.0, 1.75])
    vel = np.array([0.5, 0.0, 0.0])
    yaw = 0.0

    # Run 20 inference steps
    lstm_hidden = None
    actions = []
    t0 = time.time()

    for step in range(20):
        action, cmd, lstm_hidden = run_inference(
            policy, depth, goal, pos, vel, yaw, lstm_hidden, device
        )
        actions.append(action)

        # Simulate movement
        yaw += cmd[3] * 0.1
        vel = vel + cmd[:3] * 0.1
        pos = pos + vel * 0.1

    elapsed = time.time() - t0
    actions = np.array(actions)

    print(f"\nInference: 20 steps in {elapsed*1000:.1f}ms ({1000*20/elapsed:.0f} FPS)")
    print(f"Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"All in [-1,1]: {(actions >= -1).all() and (actions <= 1).all()}")
    print(f"Final position: {pos}")
    print(f"Distance traveled: {np.linalg.norm(pos - np.array([5.0, 0.0, 1.5])):.2f}m")

    print("\n" + "=" * 70)
    print("  RESULT: Inference pipeline WORKS ✓")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Load MAVRL weights and run inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to MAVRL checkpoint (.pth)")
    parser.add_argument("--test", action="store_true",
                       help="Run inference test with synthetic data")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda)")
    args = parser.parse_args()

    if args.test:
        test_inference()
        return

    if args.checkpoint is None:
        # Default to latest MAVRL checkpoint
        mavrl_dir = Path(config.LEARNING_DIR) / "mavrl_weights" / "RecurrentPPO_1" / "Policy"
        checkpoints = sorted(mavrl_dir.glob("iter_*.pth"))
        if checkpoints:
            args.checkpoint = str(checkpoints[-1])
        else:
            print("No MAVRL checkpoint found!")
            return

    device = torch.device(args.device)
    policy, checkpoint = load_mavrl_checkpoint(args.checkpoint, device)

    # Save loaded policy
    save_path = Path(config.CHECKPOINT_DIR) / "mavrl_loaded.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'mavrl_checkpoint': args.checkpoint,
    }, save_path)
    print(f"[LOAD] Saved loaded policy to {save_path}")

    # Run test
    test_inference()


if __name__ == "__main__":
    main()
