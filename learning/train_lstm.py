#!/usr/bin/env python3
"""
Stage C+: Train LSTM on depth reconstruction (offline, no env).

MAVRL: train_lstm_without_env.py
- Loads VAE weights (encoder frozen)
- Trains LSTM to predict future depth from sequence of latent vectors
- Uses reconstruction loss (past + current + future depth)

Usage:
    python3 learning/train_lstm.py --epochs 2000
    python3 learning/train_lstm.py --recon 0 0 1  # future only
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config
from policy import RecurrentPolicy
from vae import DepthVAE


class LSTMDepthDataset(Dataset):
    """
    Dataset for LSTM training from collected depth sequences.
    Each sample: sequence of depth images + LSTM hidden states.
    """

    def __init__(self, data_dir, seq_len=10):
        data_dir = Path(data_dir)
        data_file = data_dir / "data.npz"

        if not data_file.exists():
            raise FileNotFoundError(f"No data at {data_file}. Run Stage B first!")

        data = np.load(data_file)
        self.images = data['images']  # (N, 256, 256) uint8
        self.states = data['states']  # (N, 7) float32
        self.seq_len = seq_len

        # Split into sequences
        n = len(self.images)
        self.n_sequences = n // seq_len
        print(f"[LSTM Dataset] {n} samples → {self.n_sequences} sequences of {seq_len}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        images = self.images[start:end].astype(np.float32) / 255.0  # (seq_len, 256, 256)
        images = images[:, np.newaxis, :, :]  # (seq_len, 1, 256, 256)
        states = self.states[start:end]  # (seq_len, 7)

        return torch.FloatTensor(images), torch.FloatTensor(states)


class LSTMReconstructor(nn.Module):
    """
    LSTM-based depth reconstructor.
    Matches MAVRL: LSTM predicts future depth from sequence of latent vectors.

    Architecture (MAVRL train_lstm_without_env):
        latent(t) → LSTM → mu_linear → split into [past, current, future]
        → Decoder each → reconstructed depth
    """

    def __init__(self, policy, vae, recon_members=[False, False, True]):
        """
        Args:
            policy: RecurrentPolicy with encoder + LSTM
            vae: Trained VAE (for decoder)
            recon_members: [past, current, future] — which to reconstruct
        """
        super().__init__()
        self.encoder = policy.encoder  # frozen
        self.lstm = policy.lstm
        self.recon_members = recon_members

        # mu_linear: LSTM output → 3 × latent_dim (for past, current, future)
        latent_dim = config.FEATURES_DIM
        self.mu_linear = nn.Linear(policy.lstm_hidden, 3 * latent_dim)

        # Decoder from VAE
        self.decoder = vae.decoder
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

        # Freeze encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, images, states, lstm_hidden=None):
        """
        Forward through encoder + LSTM + reconstruction.

        Args:
            images: (batch, seq_len, 1, 256, 256)
            states: (batch, seq_len, 7)
            lstm_hidden: optional

        Returns:
            reconstruction: list of (batch, 1, 256, 256) for each recon member
            n_seq: batch size
            new_hidden: LSTM hidden state
        """
        B, T = images.shape[:2]

        # 1. Encode images (frozen encoder)
        with torch.no_grad():
            img_flat = images.reshape(B * T, 1, 256, 256)
            features = self.encoder(img_flat)  # (B*T, 64)

        # 2. LSTM input: features ONLY (matching MAVRL — state NOT in LSTM)
        lstm_in = features.reshape(B, T, -1)  # (B, T, 64)

        # 3. Run LSTM
        lstm_out, new_hidden = self.lstm(lstm_in, lstm_hidden)  # (B, T, 256)

        # 4. Predict 3 latent vectors: [past, current, future]
        pred = self.mu_linear(lstm_out)  # (B, T, 3*64)
        pred = pred.reshape(B * T, 3, config.FEATURES_DIM)

        # 5. Decode each
        reconstruction = []
        for i in range(3):
            if self.recon_members[i]:
                recon = self.decoder(pred[:, i])  # (B*T, 1, 256, 256)
                recon = recon.reshape(B, T, 1, 256, 256)
                reconstruction.append(recon)
            else:
                reconstruction.append(None)

        return reconstruction, B, new_hidden

    @property
    def state_dim(self):
        return config.STATE_DIM


def reconstruction_loss(reconstruction, target_images, recon_members):
    """
    Compute reconstruction loss for LSTM training.
    MAVRL: MSE loss for each reconstructed member.
    """
    total_loss = 0.0
    n_recon = 0

    for i, recon in enumerate(reconstruction):
        if recon is not None and recon_members[i]:
            # Shift target: past=target[:-2], current=target[1:-1], future=target[2:]
            if i == 0:  # past
                target = target_images[:, :-2]
                recon = recon[:, 1:-1] if recon.shape[1] > 2 else recon
            elif i == 1:  # current
                target = target_images[:, 1:-1]
                recon = recon[:, 1:-1] if recon.shape[1] > 2 else recon
            else:  # future
                target = target_images[:, 2:]
                recon = recon[:, :-2] if recon.shape[1] > 2 else recon

            min_len = min(recon.shape[1], target.shape[1])
            if min_len > 0:
                loss = F.mse_loss(recon[:, :min_len], target[:, :min_len], reduction='sum')
                total_loss += loss
                n_recon += 1

    return total_loss / max(n_recon, 1)


def train_lstm(args):
    """
    Train LSTM on depth reconstruction (MAVRL Stage C+).
    """
    print("=" * 60)
    print("[STAGE C+] Training LSTM on Depth Reconstruction")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[LSTM] Device: {device}")

    # 1. Load trained VAE
    vae_path = Path(config.DATA_DIR) / "vae" / "best.tar"
    if not vae_path.exists():
        print(f"[LSTM] No VAE found at {vae_path}")
        print("[LSTM] Run Stage C first!")
        return None

    vae = DepthVAE(latent_dim=config.FEATURES_DIM)
    vae_checkpoint = torch.load(vae_path, map_location=device)
    vae.load_state_dict(vae_checkpoint['state_dict'])
    vae.to(device)
    vae.eval()
    print(f"[LSTM] Loaded VAE from {vae_path} (epoch {vae_checkpoint.get('epoch', '?')})")

    # 2. Create policy with frozen encoder
    from policy import RecurrentPolicy
    policy = RecurrentPolicy(
        features_dim=config.FEATURES_DIM,
        lstm_hidden=config.LSTM_HIDDEN_SIZE,
        act_dim=config.ACTION_DIM,
        states_dim=config.STATE_DIM,
    ).to(device)

    # Load Stage A weights (LSTM + MLP)
    stage_a_path = Path(config.CHECKPOINT_DIR) / "stage_a_final.pth"
    if stage_a_path.exists():
        checkpoint = torch.load(stage_a_path, map_location=device)
        policy_state = policy.state_dict()
        loaded = 0
        for key in checkpoint['policy_state_dict']:
            if key in policy_state and not key.startswith('encoder.'):
                if checkpoint['policy_state_dict'][key].shape == policy_state[key].shape:
                    policy_state[key] = checkpoint['policy_state_dict'][key]
                    loaded += 1
        policy.load_state_dict(policy_state)
        print(f"[LSTM] Loaded {loaded} non-encoder weights from Stage A")
    else:
        print("[LSTM] No Stage A checkpoint, using random weights")

    # 3. Create reconstructor
    reconstructor = LSTMReconstructor(policy, vae, recon_members=args.recon).to(device)

    # 4. Load data
    data_dir = Path(config.DATA_DIR) / "lstm_dataset"
    if not (data_dir / "data.npz").exists():
        print(f"[LSTM] No data at {data_dir}/data.npz. Run Stage B first!")
        return None

    dataset = LSTMDepthDataset(data_dir, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 5. Optimizer (only LSTM + mu_linear, encoder frozen)
    trainable_params = list(reconstructor.lstm.parameters()) + \
                       list(reconstructor.mu_linear.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # 6. Check for resume
    save_dir = Path(config.DATA_DIR) / "lstm"
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_loss = float('inf')

    if args.resume:
        # Try to find last checkpoint
        last_ckpt = sorted(save_dir.glob("lstm_epoch_*.pth"))
        if last_ckpt:
            resume_path = last_ckpt[-1]
            print(f"[LSTM] Resuming from {resume_path.name}")
            ckpt = torch.load(resume_path, map_location=device)
            reconstructor.lstm.load_state_dict(ckpt['lstm_state_dict'])
            reconstructor.mu_linear.load_state_dict(ckpt['mu_linear_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            print(f"[LSTM] Resumed from epoch {ckpt['epoch']}, continuing from {start_epoch}")

        # Also load best loss
        best_ckpt = save_dir / "best_lstm.tar"
        if best_ckpt.exists():
            best_ckpt_data = torch.load(best_ckpt, map_location=device)
            best_loss = best_ckpt_data.get('loss', float('inf'))
            print(f"[LSTM] Best loss so far: {best_loss:.4f}")

    # 7. Training loop
    remaining = args.epochs - start_epoch + 1
    print(f"[LSTM] Training for {remaining} epochs (from {start_epoch} to {args.epochs})")
    print(f"[LSTM] Recon members: past={args.recon[0]}, current={args.recon[1]}, future={args.recon[2]}")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        reconstructor.train()
        total_loss = 0.0
        n_batches = 0

        for images, states in dataloader:
            images = images.to(device)  # (1, seq_len, 1, 256, 256)
            states = states.to(device)  # (1, seq_len, 7)

            optimizer.zero_grad()

            # Forward through encoder + LSTM + reconstruction
            reconstruction, n_seq, _ = reconstructor(images, states)

            # Compute loss
            target_images = images.squeeze(0)  # (seq_len, 1, 256, 256)
            loss = reconstruction_loss(reconstruction, target_images.unsqueeze(0), args.recon)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - start_time

        if epoch % 10 == 0:
            print(f"[LSTM] Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'lstm_state_dict': reconstructor.lstm.state_dict(),
                'mu_linear_state_dict': reconstructor.mu_linear.state_dict(),
                'loss': best_loss,
                'recon_members': args.recon,
            }, save_dir / "best_lstm.tar")

        # Save periodic
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'lstm_state_dict': reconstructor.lstm.state_dict(),
                'mu_linear_state_dict': reconstructor.mu_linear.state_dict(),
                'loss': avg_loss,
            }, save_dir / f"lstm_epoch_{epoch:05d}.pth")

    print(f"[LSTM] Training complete. Best loss: {best_loss:.6f}")
    print(f"[LSTM] Saved to {save_dir / 'best_lstm.tar'}")
    return save_dir / "best_lstm.tar"


def load_trained_lstm(policy, lstm_path, device='cpu'):
    """
    Load trained LSTM weights into policy.
    Used in Stage D.
    """
    lstm_path = Path(lstm_path)
    if not lstm_path.exists():
        print(f"[LSTM] No LSTM checkpoint at {lstm_path}")
        return False

    checkpoint = torch.load(lstm_path, map_location=device)

    policy_state = policy.state_dict()
    loaded = 0

    # Load LSTM weights
    if 'lstm_state_dict' in checkpoint:
        for key in checkpoint['lstm_state_dict']:
            policy_key = f'lstm.{key}'
            if policy_key in policy_state:
                if checkpoint['lstm_state_dict'][key].shape == policy_state[policy_key].shape:
                    policy_state[policy_key] = checkpoint['lstm_state_dict'][key]
                    loaded += 1

    print(f"[LSTM] Loaded {loaded} LSTM weights from {lstm_path}")
    policy.load_state_dict(policy_state)
    return True


def main():
    parser = argparse.ArgumentParser(description="Train LSTM on depth reconstruction")
    parser.add_argument("--epochs", type=int, default=2000,
                       help="Training epochs (MAVRL: 2000)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=10,
                       help="Sequence length for LSTM")
    parser.add_argument("--save-freq", type=int, default=100,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--recon", nargs='+', type=int, default=[0, 0, 1],
                       help="Reconstruct [past, current, future] (MAVRL: [0,0,1])")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    args = parser.parse_args()

    args.recon = [bool(x) for x in args.recon]
    train_lstm(args)


if __name__ == "__main__":
    main()
