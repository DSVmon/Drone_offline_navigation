"""
CNN + LSTM + Actor-Critic Policy for MAVRL-style drone navigation.

Architecture (matching MAVRL exactly):
    Depth Map (256×256) → CNN Encoder (6 layers) → 64-dim latent
    State (7-dim) → FC → 64-dim
    [latent, state_enc] → LSTM (256 hidden) → Actor/Critic

Based on MAVRL (TU Delft, IEEE RA-L 2025).
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class DepthEncoder(nn.Module):
    """
    CNN encoder for depth maps 256×256 → 64-dim latent.
    Matches MAVRL architecture exactly:
        conv1(1→8,4×4) → conv2(8→16,4×4) → conv3(16→32,4×4)
        → conv4(32→64,4×4) → conv5(64→128,4×4) → conv6(128→256,4×4)
        → flatten(1024) → fc_mu(1024→64) + fc_logsigma(1024→64)
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
        # 256 × 2 × 2 = 1024
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logsigma = nn.Linear(1024, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 256, 256) grayscale depth map
        Returns:
            mu: (batch, latent_dim)
            logsigma: (batch, latent_dim)
        """
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))
        h = torch.relu(self.conv4(h))
        h = torch.relu(self.conv5(h))
        h = torch.relu(self.conv6(h))
        h = self.flatten(h)
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


class RecurrentPolicy(nn.Module):
    """
    CNN + LSTM + Actor-Critic policy.
    Matches MAVRL architecture.
    """

    def __init__(self, features_dim=64, lstm_hidden=256, act_dim=4):
        super().__init__()

        self.features_dim = features_dim
        self.lstm_hidden = lstm_hidden
        self.act_dim = act_dim

        # Image encoder (matches MAVRL VAE encoder)
        self.encoder = DepthEncoder(features_dim)

        # State encoder
        self.state_fc = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=features_dim + 64,  # latent + state_enc
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.action_mean = nn.Linear(256, act_dim)
        self.action_log_std = nn.Parameter(torch.full((act_dim,), -0.5))  # MAVRL: log_std_init=-0.5

        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self._lstm_hidden_shape = (1, 1, lstm_hidden)

    def _encode_image(self, image_flat):
        """Encode flattened image batch."""
        mu, _ = self.encoder(image_flat)
        return mu

    def forward(self, image, state, lstm_hidden=None):
        """
        Forward pass through the full network.

        Args:
            image: (batch, seq, 1, 256, 256) or (batch*seq, 1, 256, 256)
            state: (batch, seq, 7)
            lstm_hidden: ((num_layers, batch, hidden), (num_layers, batch, hidden)) or None

        Returns:
            action_mean: (batch, seq, act_dim)
            action_std: (batch, seq, act_dim)
            value: (batch, seq, 1)
            new_hidden: ((num_layers, batch, hidden), (num_layers, batch, hidden))
        """
        B, T = image.shape[:2]

        # Encode images: reshape to (B*T, 1, 256, 256)
        img_flat = image.reshape(B * T, 1, 256, 256)
        latent = self._encode_image(img_flat)  # (B*T, features_dim)

        # Encode state: reshape to (B*T, 7)
        state_flat = state.reshape(B * T, 7)
        state_enc = self.state_fc(state_flat)  # (B*T, 64)

        # Combine features
        features = torch.cat([latent, state_enc], dim=-1)  # (B*T, features_dim+64)
        features = features.reshape(B, T, -1)

        # LSTM
        if lstm_hidden is None:
            lstm_out, new_hidden = self.lstm(features)
        else:
            lstm_out, new_hidden = self.lstm(features, lstm_hidden)

        # Actor
        actor_features = self.actor(lstm_out)
        action_mean = self.action_mean(actor_features)
        action_std = self.action_log_std.exp().expand_as(action_mean)

        # Critic
        value = self.critic(lstm_out)

        return action_mean, action_std, value, new_hidden

    def get_initial_hidden(self, batch_size=1, device='cpu'):
        hidden = (
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
        )
        return hidden

    def predict(self, image, state, lstm_hidden=None, deterministic=True):
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if state.dim() == 2:
                state = state.unsqueeze(0)

            mean, std, value, new_hidden = self.forward(image, state, lstm_hidden)
            mean = mean[:, -1]
            std = std[:, -1]

            if deterministic:
                action = torch.tanh(mean)
            else:
                dist = Normal(mean, std)
                raw_action = dist.sample()
                action = torch.tanh(raw_action)

            return action.cpu().numpy(), new_hidden

    def evaluate_actions(self, image, state, actions, lstm_hidden=None):
        mean, std, value, _ = self.forward(image, state, lstm_hidden)

        mean_flat = mean.reshape(-1, self.act_dim)
        std_flat = std.reshape(-1, self.act_dim)
        value_flat = value.reshape(-1)

        dist = Normal(mean_flat, std_flat)
        actions_flat = actions.reshape(-1, self.act_dim)

        log_prob = dist.log_prob(actions_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return value_flat, log_prob, entropy


class MultiInputLstmPolicy(nn.Module):
    """Wrapper for SB3 compatibility with Dict observation space."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 features_dim=64, lstm_hidden=256, n_lstm_layers=1,
                 **kwargs):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule

        act_dim = action_space.shape[0]

        self.policy = RecurrentPolicy(
            features_dim=features_dim,
            lstm_hidden=lstm_hidden,
            act_dim=act_dim,
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def forward(self, obs, deterministic=False):
        image = obs['image']
        state = obs['state']

        if image.dim() == 4:
            image = image.unsqueeze(1)
        if state.dim() == 2:
            state = state.unsqueeze(1)

        mean, std, value, _ = self.policy.forward(image, state)

        mean = mean[:, -1]
        std = std[:, -1]
        value = value[:, -1]

        if deterministic:
            action = torch.tanh(mean)
        else:
            dist = Normal(mean, std)
            raw_action = dist.sample()
            action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        return action, value, log_prob

    def predict(self, obs, lstm_hidden=None, deterministic=True):
        image = obs['image']
        state = obs['state']

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if state.dim() == 2:
            state = state.unsqueeze(0)

        action, new_hidden = self.policy.predict(
            image, state, lstm_hidden, deterministic
        )
        return action, new_hidden

    def get_initial_hidden(self, batch_size=1, device='cpu'):
        return self.policy.get_initial_hidden(batch_size, device)

    def evaluate_actions(self, obs, actions, lstm_hidden=None):
        image = obs['image']
        state = obs['state']

        values, log_prob, entropy = self.policy.evaluate_actions(
            image, state, actions, lstm_hidden
        )
        return values, log_prob, entropy

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
