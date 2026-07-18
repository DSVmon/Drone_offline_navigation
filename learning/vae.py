"""
Variational AutoEncoder (VAE) for depth map pre-training.

Matches MAVRL architecture exactly:
    Encoder: (1, 256, 256) → 6 Conv layers → 64-dim latent
    Decoder: 64-dim → 6 DeConv layers → (1, 256, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

import config


class DepthEncoder(nn.Module):
    """
    VAE Encoder matching MAVRL architecture.
    conv1(1→8) → conv2(8→16) → conv3(16→32) → conv4(32→64)
    → conv5(64→128) → conv6(128→256) → fc_mu(1024→64) + fc_logsigma(1024→64)
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 4, stride=4, padding=0)    # 256→64
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)   # 64→32
        self.conv3 = nn.Conv2d(16, 32, 4, stride=2, padding=1)  # 32→16
        self.conv4 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 16→8
        self.conv5 = nn.Conv2d(64, 128, 4, stride=2, padding=1) # 8→4
        self.conv6 = nn.Conv2d(128, 256, 4, stride=2, padding=1)# 4→2
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logsigma = nn.Linear(1024, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logsigma = self.fc_logsigma(h)
        return mu, logsigma

    def encode(self, x, deterministic=True):
        mu, logsigma = self.forward(x)
        if deterministic:
            return mu
        std = logsigma.exp()
        eps = torch.randn_like(std)
        return mu + eps * std


class DepthDecoder(nn.Module):
    """
    VAE Decoder matching MAVRL architecture.
    fc1(64→1024) → deconv1(1024→128) → deconv2(128→64) → deconv3(64→32)
    → deconv4(32→16) → deconv5(16→8) → deconv6(8→1)
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=0)
        self.deconv5 = nn.ConvTranspose2d(16, 8, 5, stride=2, padding=0)
        self.deconv6 = nn.ConvTranspose2d(8, 1, 4, stride=4, padding=0)

    def forward(self, z):
        h = self.fc1(z)
        h = h.view(-1, 1024, 1, 1)
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


class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x_recon, x, mu, logsigma):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logsigma - mu.pow(2) - logsigma.exp())
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss


def load_mavrl_vae_weights(checkpoint_path, device='cpu'):
    """
    Load pre-trained MAVRL VAE weights.
    
    Args:
        checkpoint_path: path to vae/best.tar from MAVRL
        device: device to load weights to
        
    Returns:
        encoder: trained DepthEncoder with MAVRL weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = DepthEncoder(latent_dim=64)
    
    # Extract encoder weights from MAVRL checkpoint
    mavrl_state = checkpoint['state_dict']
    encoder_state = encoder.state_dict()
    
    # Map MAVRL keys to our keys
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


def load_mavrl_policy_weights(checkpoint_path, policy, device='cpu'):
    """
    Load pre-trained MAVRL policy weights (encoder only).
    
    Args:
        checkpoint_path: path to RecurrentPPO_1/Policy/iter_XXXXX.pth
        policy: RecurrentPolicy or MultiInputLstmPolicy to load into
        device: device to load weights to
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[VAE] Cannot load PPO weights: {e}")
        print("[VAE] PPO weights require 'gym' module. Use VAE weights instead.")
        return False
    
    if 'state_dict' not in checkpoint:
        print("[VAE] PPO checkpoint format not recognized")
        return False
    
    mavrl_state = checkpoint['state_dict']
    policy_state = policy.state_dict()
    
    # Map MAVRL encoder keys to policy encoder keys
    key_map = {
        'features_extractor.conv1.weight': 'policy.encoder.conv1.weight',
        'features_extractor.conv1.bias': 'policy.encoder.conv1.bias',
        'features_extractor.conv2.weight': 'policy.encoder.conv2.weight',
        'features_extractor.conv2.bias': 'policy.encoder.conv2.bias',
        'features_extractor.conv3.weight': 'policy.encoder.conv3.weight',
        'features_extractor.conv3.bias': 'policy.encoder.conv3.bias',
        'features_extractor.conv4.weight': 'policy.encoder.conv4.weight',
        'features_extractor.conv4.bias': 'policy.encoder.conv4.bias',
        'features_extractor.conv5.weight': 'policy.encoder.conv5.weight',
        'features_extractor.conv5.bias': 'policy.encoder.conv5.bias',
        'features_extractor.conv6.weight': 'policy.encoder.conv6.weight',
        'features_extractor.conv6.bias': 'policy.encoder.conv6.bias',
        'features_extractor.linear.weight': 'policy.encoder.fc_mu.weight',
        'features_extractor.linear.bias': 'policy.encoder.fc_mu.bias',
        'features_extractor.fc_logsigma.weight': 'policy.encoder.fc_logsigma.weight',
        'features_extractor.fc_logsigma.bias': 'policy.encoder.fc_logsigma.bias',
    }
    
    transferred = 0
    for mavrl_key, our_key in key_map.items():
        if mavrl_key in mavrl_state and our_key in policy_state:
            if mavrl_state[mavrl_key].shape == policy_state[our_key].shape:
                policy_state[our_key] = mavrl_state[mavrl_key]
                transferred += 1
    
    policy.load_state_dict(policy_state)
    print(f"[VAE] Loaded MAVRL encoder into policy: {transferred}/{len(key_map)} weights")
    return True


def train_vae(data_path, save_path=None, epochs=100, batch_size=64, lr=1e-3, beta=1.0):
    """Train VAE on depth data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[VAE] Device: {device}")

    data = np.load(data_path)
    images = data['images'].astype(np.float32) / 255.0
    images = images[:, np.newaxis, :, :]  # (N, 1, 256, 256)
    print(f"[VAE] Loaded {len(images)} depth images")

    dataset = TensorDataset(torch.FloatTensor(images))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = DepthVAE(latent_dim=config.FEATURES_DIM).to(device)
    criterion = VAELoss(beta=beta)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_loss = float('inf')
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        n_batches = 0

        for batch in dataloader:
            x = batch[0].to(device)
            x_recon, mu, logsigma = vae(x)
            loss, _, _ = criterion(x_recon, x, mu, logsigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[VAE] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'encoder_state_dict': vae.encoder.state_dict(),
                    'decoder_state_dict': vae.decoder.state_dict(),
                    'latent_dim': config.FEATURES_DIM,
                    'loss': best_loss,
                }, save_path)

    print(f"[VAE] Training complete. Best loss: {best_loss:.6f}")
    return vae
