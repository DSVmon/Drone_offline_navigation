---
name: mavrl-reference
description: Fetch and compare MAVRL source code from TU Delft GitHub for alignment verification. Covers key files for architecture, config, training, and inference.
---

# MAVRL Reference Fetch

Fetch raw source files from the MAVRL repository (TU Delft, IEEE RA-L 2025) for architecture comparison and alignment verification.

## Repository

- **URL**: `https://github.com/tudelft/mavrl`
- **Raw base**: `https://raw.githubusercontent.com/tudelft/mavrl/main/`

## Key Files to Reference

### Architecture & Config
| File | Purpose | Key Values |
|------|---------|------------|
| `avoider_vel_cmd.py` | Main inference agent | sim_dt=0.1, depth normalization, action conversion |
| `config.yaml` | Hyperparameters | All training config, camera params, reward coefficients |
| `recurrent/policies.py` | RecurrentPolicy + forward_rnn | CNN encoder → LSTM → MLP, two-stage forward |
| `recurrent/recurrent_ppo.py` | RecurrentPPO | Latent caching, rollout buffer, GAE |

### Training Pipeline
| File | Purpose |
|------|---------|
| `train.py` | Main training entry point |
| `trainvae.py` | VAE standalone training |
| `train_lstm.py` | LSTM offline training |
| `recurrent/buffer.py` | Rollout buffer with latent caching |
| `recurrent/ppo.py` | PPO algorithm core |

### Environment
| File | Purpose |
|------|---------|
| `avoid_vision_envs.py` | Unity/GPU environment wrapper |
| `avoid_vision_envs.h` | C++ env with SGM stereo |

## Fetch Commands

Use `webfetch` tool with format="text" for each file:

```
webfetch: https://raw.githubusercontent.com/tudelft/mavrl/main/avoider_vel_cmd.py
webfetch: https://raw.githubusercontent.com/tudelft/mavrl/main/config.yaml
webfetch: https://raw.githubusercontent.com/tudelft/mavrl/main/recurrent/policies.py
webfetch: https://raw.githubusercontent.com/tudelft/mavrl/main/recurrent/recurrent_ppo.py
webfetch: https://raw.githubusercontent.com/tudelft/mavrl/main/train.py
webfetch: https://raw.githubusercontent.com/tudelft/mavrl/main/trainvae.py
webfetch: https://raw.githubusercontent.com/tudelft/mavrl/main/train_lstm.py
```

## Architecture Comparison Quick Reference

| Component | Our Project | MAVRL |
|-----------|-------------|-------|
| **Depth source** | StereoBM (CPU) | SGM (CUDA GPU) |
| **Encoder** | 6-layer CNN, stride=2, 64-dim | Same (identical architecture) |
| **LSTM** | 256 hidden, input=71 (64+7) | Same, states_dim=0 (state NOT fed to LSTM) |
| **MLP** | input=263 (256+7) | Same (MlpExtractor) |
| **Action** | 4-dim (ax, ay, az, yaw_rate) | Same |
| **Forward pass** | forward_rnn → latents → forward | Same two-stage pattern |
| **Normalization** | uint8 → float32/255.0 | SB3 preprocess_obs(normalize_images=True) |
| **VAE** | train_vae() standalone | trainvae.py standalone |
| **LSTM training** | train_lstm_from_dataset() | train_lstm.py offline |
| **PPO** | RecurrentPPO custom | RecurrentPPO custom |
| **Data format** | .npz, float32 | .pt, int |

## Key Structural Notes

1. **MAVRL `forward_rnn()`** is the critical forward pass:
   - Extracts CNN features from image
   - Runs LSTM on features only (state_dim=0)
   - Concatenates LSTM output with state observation
   - Returns [latent_pi, latent_vf] for MLP

2. **MAVRL `evaluate_actions()`** is decoupled from observation encoding:
   - Takes pre-computed `latent_lstm_pi/vf` directly
   - Allows caching latent states in rollout buffer

3. **MAVRL uses TWO stereo cameras + SGM GPU** (not ground truth depth):
   - From `avoid_vision_envs.h`: `std::shared_ptr<RGBCamera> rgb_camera_, right_rgb_camera_`
   - RGB camera (`rgb_camera` config with `r_BC: [0.0, 0.0, 90.0]`) is for visualization only

4. **MAVRL LR decays to 0**: `1e-4 * progress_remaining`. Our LEARNING_RATE_END should be 0.0.
