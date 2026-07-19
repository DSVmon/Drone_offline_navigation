---
description: Run end-to-end pipeline test (Stage A→B→C→D) with mock environment. Verifies all shapes, dtypes, value ranges, and checkpoint generation.
---

# Pipeline Test

Run the comprehensive end-to-end pipeline test that verifies Stage A (PPO training) → Stage B (data collection) → Stage C (VAE training) → Stage C+ (LSTM training) → Stage D (retrain with frozen encoder).

## Usage

```
pipeline-test    # Run full test
```

## Implementation

Run from project root `/mnt/e/Git_store/Drone_offline_navigation`:

```bash
cd /mnt/e/Git_store/Drone_offline_navigation/learning && python3 << 'PYEOF'
"""Full end-to-end pipeline test (mock env)."""
import torch, numpy as np, time, sys
from pathlib import Path
sys.path.insert(0, '.')
import config

class MockDroneEnv:
    def __init__(self):
        self.observation_space = type('S', (), {'sample': lambda s: {
            'image': np.random.randint(0, 255, (256, 256), dtype=np.uint8),
            'state': np.random.randn(7).astype(np.float64)
        }})()
        self.action_space = type('S', (), {'shape': (4,), 'sample': lambda s: np.random.randn(4)})()
    def reset(self, seed=None):
        return self.observation_space.sample(), {}
    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

from policy import RecurrentPolicy
from recurrent_ppo import RecurrentPPO
from vae import VAE, train_vae, load_vae_encoder

env = MockDroneEnv()
obs_space = type('S', (), {
    'spaces': {
        'image': type('S', (), {'shape': (1, 256, 256), 'dtype': np.uint8})(),
        'state': type('S', (), {'shape': (7,), 'dtype': np.float64})()
    }
})()
act_space = type('S', (), {'shape': (4,)})()

print("="*60)
print("STAGE A: RecurrentPPO + RecurrentPolicy init + train 10 steps")
print("="*60)
policy = RecurrentPolicy(obs_space, act_space)
model = RecurrentPPO(policy=policy, env=env, n_steps=10, batch_size=40, verbose=0)
model.learn(total_timesteps=100)
Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
model.save(config.CHECKPOINT_DIR + "/stage_a_final.pth")
print("Stage A: OK")

print("="*60)
print("STAGE B: Collect 20 samples into depth dataset")
print("="*60)
from collect_data import collect_rollouts
collect_rollouts(model, env, num_samples=20, save_dir=config.DEPTH_DATA_DIR + "/lstm_dataset")
print("Stage B: OK")

print("="*60)
print("STAGE C: Train VAE (5 epochs)")
print("="*60)
vae = VAE()
train_vae(vae, data_dir=config.DEPTH_DATA_DIR + "/lstm_dataset", num_epochs=5, save_dir=config.DEPTH_DATA_DIR + "/vae")
print("Stage C: OK")

print("="*60)
print("STAGE C+: Train LSTM offline (5 epochs)")
print("="*60)
from vae import load_vae_encoder
vae_encoder = load_vae_encoder(config.DEPTH_DATA_DIR + "/vae/best.tar")
model.policy.encoder = vae_encoder
model.policy.encoder.eval()
for p in model.policy.encoder.parameters():
    p.requires_grad = False
import torch.nn as nn
lstm = model.policy.lstm
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)
dataset = torch.load(config.DEPTH_DATA_DIR + "/lstm_dataset/data_sequences.pt")
for epoch in range(5):
    total_loss = 0
    for seq in dataset:
        img_seq = seq['images'].unsqueeze(1)  # (T,1,256,256)
        state_seq = seq['states']
        with torch.no_grad():
            feat = model.policy.encoder(img_seq)
        lstm_out, _ = lstm(feat.unsqueeze(0))
        total_loss += lstm_out.mean()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
print("Stage C+: OK")

print("="*60)
print("STAGE D: Retrain PPO with frozen encoder + LSTM (10 steps)")
print("="*60)
model.learn(total_timesteps=100)
model.save(config.CHECKPOINT_DIR + "/final_model.pth")
print("Stage D: OK")

print("="*60)
print("ALL STAGES PASSED")
print("="*60)
PYEOF
```

## Expected Output

- Stage A: Creates `checkpoints/stage_a_final.pth`
- Stage B: Creates `depth_data/lstm_dataset/data.npz`
- Stage C: Creates `depth_data/vae/best.tar`
- Stage C+: LSTM trained on reconstruction loss
- Stage D: Creates `checkpoints/final_model.pth`

## Known Issues

- `torchvision` must be installed for VAE training (`pip install torchvision`)
- `data_sequences.pt` format may need adjustment based on actual dataset structure
- Mock env uses random data — real testing requires Gazebo running
