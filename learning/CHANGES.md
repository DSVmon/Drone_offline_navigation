# Changes Summary - Training Pipeline Overhaul

## Critical Bug Fixes

### 1. Observation Dimension Mismatch (FIXED)
- **Before**: inference_node.py used 13-dim obs, drone_env.py used 19-dim
- **After**: All files use 14-dim obs: `stereo[5] + x, y, z, sin_yaw, cos_yaw, vx, vz, roll, pitch`
- **Files**: drone_env.py, inference_node.py, collect_expert.py

### 2. Z-axis Control (FIXED)
- **Before**: hover_correction term interfered with policy control
- **After**: Pure teleport: `target_z = current_z + vz * DT`
- **File**: drone_env.py

### 3. Expert Data Cleanup
- **Before**: 500 batches (250 x 13-dim, 250 x 19-dim)
- **After**: 250 batches (19-dim only, ~50K samples)
- **Note**: Need to recollect with 14-dim format for new training

## Reward Function Overhaul

### New Reward Components
| Component | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| r_progress | +1.0 | +0.7 | +0.5 |
| r_proximity | -0.05 | -0.10 | -0.15 |
| r_speed | +0.03 | +0.03 | +0.03 |
| r_survive | +0.01 | +0.01 | +0.01 |
| r_collision | -5.0 | -5.0 | -5.0 |
| r_out_of_bounds | -10.0 | -10.0 | -10.0 |
| r_timeout | -3.0 | -3.0 | -3.0 |
| r_completion | +50.0 | +50.0 | +50.0 |

### Key Changes
- **Added**: r_progress (progress along cave axis) - DOMINANT signal
- **Reduced**: proximity penalty from -0.3 to -0.15
- **Reduced**: collision penalty from -15 to -5
- **Changed**: proximity threshold from 1.0m to 0.5m
- **Reduced**: episode timeout from 300s to 120s

## PPO Hyperparameters

| Parameter | Before | After |
|-----------|--------|-------|
| TOTAL_TIMESTEPS | 1M | 2M |
| LEARNING_RATE | 3e-4 | 1e-4 |
| N_STEPS | 2048 | 4096 |
| BATCH_SIZE | 64 | 128 |
| N_EPOCHS | 10 | 20 |
| GAMMA | 0.99 | 0.995 |
| ENT_COEF | 0.001 | 0.01 |
| net_arch | [128, 128] | [256, 256] |
| activation_fn | Tanh | ReLU |

## New Files

### learning/curriculum.py
- Curriculum learning manager
- Tracks stage progression based on success rate
- Auto-advances when success > 80%
- Auto-regresses when success < 30%

### scripts/straight_cave.py
- Stage 1 cave generator
- 50m straight tunnel, 6m wide
- No turns, no obstacles

### scripts/gentle_cave.py
- Stage 2 cave generator
- 80m tunnel with ±30° turns
- Light stalactites every 20 segments

### learning/monitor.py
- Training monitor and controller
- Commands: status, pause, resume, adjust

## Training Strategy

### Curriculum Learning
1. **Stage 1 (0-500K)**: Straight cave, learn to fly forward
2. **Stage 2 (500K-1M)**: Gentle turns, learn to avoid walls
3. **Stage 3 (1M-2M)**: Full complexity, learn to navigate

### Reward Balance
- Stage 1: Progress dominates (20:1 ratio)
- Stage 2: Progress still dominant (7:1)
- Stage 3: Balanced (3.3:1)

## Next Steps

1. ~~Recollect expert data with 14-dim format~~
2. ~~Run BC training~~
3. ~~Start PPO training with curriculum~~
4. ~~Monitor and adjust as needed~~

## Curriculum Integration (Current)

### What was integrated
- `drone_env.py`: CurriculumManager selects cave script per stage, passes reward_overrides to compute_reward
- `train.py`: Added CurriculumCallback for periodic stage checks
- `callbacks.py`: New CurriculumCallback class — checks should_advance/should_regress every 50K steps
- `utils.py`: generate_cave() and full_reset_simulation() now accept cave_script parameter
- `reward.py`: compute_reward() accepts reward_overrides dict for dynamic coefficients
- `config.py`: Removed fixed CAVE_SCRIPT — now determined by curriculum
- `monitor.py`: Enhanced show_status() with curriculum stage details and success rate

### Training workflow
1. Stage 1 (0-500K): straight_cave.py, progress_coeff=1.0, proximity_coeff=-0.05
2. Stage 2 (500K-1M): gentle_cave.py, progress_coeff=0.7, proximity_coeff=-0.10
3. Stage 3 (1M-2M): procedural_cave.py, progress_coeff=0.5, proximity_coeff=-0.15

### How to run
```bash
# Terminal 1: Start training
cd learning && python3 train.py

# Terminal 2: Monitor
python3 learning/monitor.py status

# Terminal 3: TensorBoard
tensorboard --logdir learning/tensorboard_logs
```
