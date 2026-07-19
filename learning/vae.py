"""
Variational AutoEncoder (VAE) for depth map pre-training.

Matches MAVRL architecture exactly:
    Encoder: (1, 256, 256) -> 6 Conv layers (stride=2) -> 64-dim latent
    Decoder: 64-dim -> 6 DeConv layers -> (1, 256, 256)

Training flow (MAVRL):
    1. Collect depth data with initial policy -> saved/lstm_dataset/
    2. Train VAE on depth images -> exp_vae/vae/best.tar
    3. Load VAE encoder weights into policy for Stage D

Reference: github.com/tudelft/mavrl/trainvae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import config


class DepthEncoder(nn.Module):
    """
    VAE Encoder matching MAVRL architecture exactly.
    conv1(1->8, k=4, s=2)   -> 256->127
    conv2(8->16, k=4, s=2)  -> 127->62
    conv3(16->32, k=4, s=2) -> 62->30
    conv4(32->64, k=4, s=2) -> 30->14
    conv5(64->128, k=4, s=2)-> 14->6
    conv6(128->256, k=4, s=2)-> 6->2
    flatten(256*2*2=1024) -> fc_mu(1024->64) + fc_logsigma(1024->64)
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 4, stride=2)     # 256->127
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2)    # 127->62
        self.conv3 = nn.Conv2d(16, 32, 4, stride=2)   # 62->30
        self.conv4 = nn.Conv2d(32, 64, 4, stride=2)   # 30->14
        self.conv5 = nn.Conv2d(64, 128, 4, stride=2)  # 14->6
        self.conv6 = nn.Conv2d(128, 256, 4, stride=2) # 6->2
        # 256 * 2 * 2 = 1024
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logsigma = nn.Linear(256 * 2 * 2, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = h.view(h.size(0), -1)  # flatten: (batch, 1024)
        mu = self.fc_mu(h)
        logsigma = self.fc_logsigma(h)
        return mu, logsigma

    def encode(self, x, deterministic=True):
        """Encode with optional reparameterization."""
        mu, logsigma = self.forward(x)
        if deterministic:
            return mu
        std = logsigma.exp()
        eps = torch.randn_like(std)
        return mu + eps * std


class DepthDecoder(nn.Module):
    """
    VAE Decoder matching MAVRL architecture.
    fc1(64->1024) -> reshape(1024->256*2*2)
    -> deconv1(1024->128, k=5, s=2) -> deconv2(128->64, k=5, s=2)
    -> deconv3(64->32, k=6, s=2) -> deconv4(32->16, k=4, s=2)
    -> deconv5(16->8, k=5, s=2) -> deconv6(8->1, k=4, s=2)
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(256 * 2 * 2, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.deconv5 = nn.ConvTranspose2d(16, 8, 5, stride=2)
        self.deconv6 = nn.ConvTranspose2d(8, 1, 4, stride=2)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = h.view(-1, 256 * 2 * 2, 1, 1)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        h = F.relu(self.deconv4(h))
        h = F.relu(self.deconv5(h))
        x_recon = torch.sigmoid(self.deconv6(h))
        return x_recon


class DepthVAE(nn.Module):
    """VAE matching MAVRL architecture."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = DepthEncoder(latent_dim)
        self.decoder = DepthDecoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logsigma):
        if self.training:
            std = logsigma.exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = self.reparameterize(mu, logsigma)
        x_recon = self.decoder(z)
        return x_recon, mu, logsigma

    def encode(self, x, deterministic=True):
        return self.encoder.encode(x, deterministic)

    def decode(self, z):
        return self.decoder(z)


# --- Dataset (MAVRL _RolloutDataset style) ---

class DepthImageDataset(Dataset):
    """
    Dataset for VAE training from collected depth images.
    Loads .npz files from lstm_dataset/ directory.

    MAVRL format: saved/lstm_dataset/ contains depth sequences.
    Our format: depth_data/lstm_dataset/data.npz with 'images' key.
    """

    def __init__(self, data_dir, transform=None, train=True, train_ratio=0.8):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Load data
        data_file = self.data_dir / "data.npz"
        if not data_file.exists():
            raise FileNotFoundError(f"No data found at {data_file}. Run Stage B first!")

        data = np.load(data_file)
        images = data['images']  # (N, 256, 256) uint8

        # Train/test split
        n = len(images)
        split = int(n * train_ratio)
        if train:
            self.images = images[:split]
        else:
            self.images = images[split:]

        print(f"[Dataset] {'Train' if train else 'Test'}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (256, 256) uint8

        # Convert to float tensor [0, 1] with channel dim
        img_tensor = torch.FloatTensor(img).unsqueeze(0) / 255.0  # (1, 256, 256)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor


# --- VAE Loss (MAVRL style) ---

def vae_loss_function(recon_x, x, mu, logsigma):
    """
    VAE loss = Reconstruction + KL divergence.
    MAVRL: BCE(recon, x) + KLD(mu, logsigma)
    """
    # Reconstruction loss (MSE, matching MAVRL)
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return BCE + KLD, BCE, KLD


# --- Training (MAVRL trainvae.py style) ---

def train_vae(data_path=None, save_path=None, epochs=100, batch_size=32,
              lr=1e-3, beta=1.0, patience=50, factor=0.5):
    """
    Train VAE on collected depth data.

    Matching MAVRL trainvae.py:
    - Uses ReduceLROnPlateau scheduler
    - Uses EarlyStopping
    - Saves best model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[VAE] Device: {device}")

    # Default paths
    if data_path is None:
        data_path = config.DATA_DIR / "lstm_dataset"
    if save_path is None:
        save_path = config.DATA_DIR / "vae" / "best.tar"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Transforms (matching MAVRL)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])

    # Create datasets
    dataset_train = DepthImageDataset(data_path, transform=transform_train, train=True)
    dataset_test = DepthImageDataset(data_path, transform=None, train=False)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    model = DepthVAE(latent_dim=config.FEATURES_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler (matching MAVRL)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=factor, patience=10
    )

    # Early stopping (matching MAVRL)
    best_loss = float('inf')
    patience_counter = 0

    print(f"[VAE] Training for {epochs} epochs, batch_size={batch_size}")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x)
            loss, _, _ = vae_loss_function(recon_batch, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(dataset_train)

        # --- Test ---
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch.to(device)
                recon_batch, mu, logvar = model(x)
                loss, _, _ = vae_loss_function(recon_batch, x, mu, logvar)
                test_loss += loss.item()
        test_loss /= len(dataset_test)

        scheduler.step(test_loss)

        # --- Logging ---
        if epoch % 10 == 0:
            print(f"[VAE] Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Test: {test_loss:.4f}")

        # --- Save best ---
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'precision': test_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, save_path)
        else:
            patience_counter += 1

        # --- Early stopping ---
        if patience_counter >= patience:
            print(f"[VAE] Early stopping at epoch {epoch} (patience={patience})")
            break

    print(f"[VAE] Training complete. Best test loss: {best_loss:.6f}")
    print(f"[VAE] Saved to {save_path}")
    return model


# --- Weight loading utilities ---

def load_mavrl_vae_weights(checkpoint_path, device='cpu'):
    """Load pre-trained MAVRL VAE weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = DepthEncoder(latent_dim=64)

    mavrl_state = checkpoint.get('state_dict', checkpoint)
    encoder_state = encoder.state_dict()

    key_map = {
        'encoder.conv1.weight': 'conv1.weight',
        'encoder.conv1.bias': 'conv1.bias',
        'encoder.conv2.weight': 'conv2.weight',
        'encoder.conv2.bias': 'conv2.bias',
        'encoder.conv3.weight': 'conv3.weight',
        'encoder.conv3.bias': 'conv3.bias',
        'encoder.conv4.weight': 'conv4.weight',
        'encoder.conv4.bias': 'conv4.bias',
        'encoder.conv5.weight': 'conv5.weight',
        'encoder.conv5.bias': 'conv5.bias',
        'encoder.conv6.weight': 'conv6.weight',
        'encoder.conv6.bias': 'conv6.bias',
        'encoder.fc_mu.weight': 'fc_mu.weight',
        'encoder.fc_mu.bias': 'fc_mu.bias',
        'encoder.fc_logsigma.weight': 'fc_logsigma.weight',
        'encoder.fc_logsigma.bias': 'fc_logsigma.bias',
    }

    transferred = 0
    for mavrl_key, our_key in key_map.items():
        if mavrl_key in mavrl_state and our_key in encoder_state:
            if mavrl_state[mavrl_key].shape == encoder_state[our_key].shape:
                encoder_state[our_key] = mavrl_state[mavrl_key]
                transferred += 1

    encoder.load_state_dict(encoder_state)
    print(f"[VAE] Loaded MAVRL encoder: {transferred}/{len(key_map)} weights transferred")
    return encoder


def load_vae_encoder(vae_path, policy, device='cpu'):
    """
    Load trained VAE encoder weights into policy.
    Used in Stage D to initialize policy encoder.
    """
    vae_path = Path(vae_path)
    if not vae_path.exists():
        print(f"[VAE] No VAE checkpoint found at {vae_path}")
        return False

    checkpoint = torch.load(vae_path, map_location=device)

    # Try different checkpoint formats
    if 'state_dict' in checkpoint:
        vae_state = checkpoint['state_dict']
    elif 'encoder_state_dict' in checkpoint:
        vae_state = checkpoint['encoder_state_dict']
    else:
        vae_state = checkpoint

    policy_state = policy.state_dict()
    transferred = 0

    # Map VAE encoder keys to policy encoder keys
    for vae_key in vae_state:
        if vae_key.startswith('encoder.'):
            policy_key = vae_key  # Already has 'encoder.' prefix
        else:
            policy_key = f'encoder.{vae_key}'

        if policy_key in policy_state:
            if vae_state[vae_key].shape == policy_state[policy_key].shape:
                policy_state[policy_key] = vae_state[vae_key]
                transferred += 1

    policy.load_state_dict(policy_state)
    print(f"[VAE] Loaded {transferred} encoder weights into policy")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train VAE on depth data")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with lstm_dataset/data.npz")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save VAE checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    train_vae(
        data_path=args.data_dir,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
