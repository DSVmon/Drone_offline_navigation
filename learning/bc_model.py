import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import config


class BCNet(nn.Module):
    """MLP for Behavior Cloning matching PPO MlpPolicy architecture."""

    def __init__(self, obs_dim, act_dim, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = config.BC_HIDDEN

        layers = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.Tanh())
            prev_size = h
        layers.append(nn.Linear(prev_size, act_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)


def load_expert_data(expert_dir=None):
    """Load all expert .npz files from the expert data directory."""
    if expert_dir is None:
        expert_dir = config.EXPERT_DIR

    expert_dir = Path(expert_dir)
    all_obs = []
    all_acts = []

    for npz_file in sorted(expert_dir.glob("*.npz")):
        data = np.load(npz_file)
        all_obs.append(data["observations"])
        all_acts.append(data["actions"])

    if not all_obs:
        raise FileNotFoundError(
            f"No expert data found in {expert_dir}. "
            "Run collect_expert.py first."
        )

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_acts, axis=0)

    # Normalize actions if they're in raw (old) format
    if actions.shape[-1] == 3 and np.any(np.abs(actions) > 1.5):
        print(f"[BC] Detected raw actions (max={np.max(np.abs(actions)):.2f}), normalizing to [-1,1]")
        actions = config.normalize_action(actions)

    return observations, actions


def train_bc(expert_dir=None, save_path=None, device=None):
    """Train a BC model on expert data with validation and LR scheduling."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if save_path is None:
        save_path = config.EXPERT_DIR / "bc_policy.pt"

    observations, actions = load_expert_data(expert_dir)
    obs_dim = observations.shape[1]
    act_dim = actions.shape[1]

    # --- Train/val split ---
    n = len(observations)
    n_val = int(n * config.BC_VAL_SPLIT)
    n_train = n - n_val
    perm = np.random.RandomState(42).permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_obs = torch.FloatTensor(observations[train_idx])
    train_act = torch.FloatTensor(actions[train_idx])
    val_obs = torch.FloatTensor(observations[val_idx])
    val_act = torch.FloatTensor(actions[val_idx])

    train_dataset = torch.utils.data.TensorDataset(train_obs, train_act)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BC_BATCH_SIZE, shuffle=True
    )

    model = BCNet(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.BC_LR)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.BC_LR_FACTOR,
        patience=config.BC_LR_PATIENCE, min_lr=config.BC_MIN_LR,
        threshold=1e-5
    )
    mse_loss = nn.MSELoss()

    print(f"[BC] Training on {n_train} samples, validating on {n_val}")
    print(f"[BC] Device: {device}")

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    max_patience = 80

    for epoch in range(config.BC_EPOCHS):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_obs, batch_act in train_loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            pred = model(batch_obs)
            loss = mse_loss(pred, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_obs.to(device))
            val_loss = mse_loss(val_pred, val_act.to(device)).item()

        lr_scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[BC] Epoch {epoch+1}/{config.BC_EPOCHS} | "
                  f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if epochs_no_improve >= max_patience:
            print(f"[BC] Early stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[BC] Model saved to {save_path} (best val loss: {best_val_loss:.6f})")

    return model


def load_bc_into_ppo(bc_model, ppo_policy):
    """Transfer BC weights into a Stable-Baselines3 PPO policy."""
    import torch

    with torch.no_grad():
        bc_state = bc_model.state_dict()

        bc_feature_keys = [k for k in bc_state.keys() if "network." in k and "network.4" not in k]
        bc_action_key = [k for k in bc_state.keys() if "network.4" in k]

        mlp_keys = [k for k in ppo_policy.mlp_extractor.state_dict().keys()
                    if "shared_net" in k]
        mlp_keys.sort()

        if len(mlp_keys) == len(bc_feature_keys):
            for bc_k, ppo_k in zip(bc_feature_keys, mlp_keys):
                bc_param = bc_state[bc_k]
                ppo_param = ppo_policy.mlp_extractor.state_dict()[ppo_k]
                if bc_param.shape == ppo_param.shape:
                    ppo_policy.mlp_extractor.state_dict()[ppo_k].copy_(bc_param)

        action_keys = list(ppo_policy.action_net.state_dict().keys())
        action_keys.sort()
        if len(action_keys) == len(bc_action_key):
            for bc_k, ppo_k in zip(bc_action_key, action_keys):
                bc_param = bc_state[bc_k]
                ppo_param = ppo_policy.action_net.state_dict()[ppo_k]
                if bc_param.shape == ppo_param.shape:
                    ppo_policy.action_net.state_dict()[ppo_k].copy_(bc_param)

        print("[BC] BC weights transferred into PPO policy "
              f"(mlp: {len(bc_feature_keys)}, action: {len(bc_action_key)})")
