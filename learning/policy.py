"""
CNN + LSTM + Actor-Critic Policy for MAVRL-style drone navigation.

Architecture (matching MAVRL exactly):
    Depth Map (256x256) -> CNN Encoder (6 conv layers, stride=2) -> 64-dim
    LSTM input: [features(64) + state(7)] = 71-dim
    MLP input:  [lstm_out(256) + state(7)] = 263-dim
    Actor: 263 -> [256,256] -> 4 (Tanh)
    Critic: 263 -> [512,512] -> 1

Based on MAVRL (TU Delft, IEEE RA-L 2025).
Reference: github.com/tudelft/mavrl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class DepthEncoder(nn.Module):
    """
    CNN encoder for depth maps 256x256 -> 64-dim latent.
    Matches MAVRL Encoder (rnn_extractor.py) exactly:
        conv1(1->8, k=4, s=2)   -> 256->128
        conv2(8->16, k=4, s=2)  -> 128->64
        conv3(16->32, k=4, s=2) -> 64->32
        conv4(32->64, k=4, s=2) -> 32->16
        conv5(64->128, k=4, s=2)-> 16->8
        conv6(128->256, k=4, s=2)-> 8->4
        flatten(256*4*4=4096) -> fc_mu(4096->64)
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

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 256, 256) grayscale depth map
        Returns:
            mu: (batch, latent_dim) - deterministic encoding
        """
        # Convert uint8 to float32 and normalize to [0,1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = h.view(h.size(0), -1)  # flatten: (batch, 1024)
        mu = self.fc_mu(h)          # (batch, 64)
        return mu


class RecurrentPolicy(nn.Module):
    """
    CNN + LSTM + Actor-Critic policy.
    Matches MAVRL architecture exactly.

    Data flow (MAVRL):
        1. features = encoder(image)           # (batch, 64)
        2. lstm_in = [features, state]         # (batch, 71)
        3. lstm_out = lstm(lstm_in)            # (batch, 256)
        4. mlp_in = [lstm_out, state]          # (batch, 263)
        5. action = actor(mlp_in)              # (batch, 4)
        6. value = critic(mlp_in)              # (batch, 1)
    """

    def __init__(self, features_dim=64, lstm_hidden=256, act_dim=4, states_dim=7):
        super().__init__()

        self.features_dim = features_dim
        self.lstm_hidden = lstm_hidden
        self.act_dim = act_dim
        self.states_dim = states_dim

        # Image encoder (matches MAVRL Encoder exactly)
        self.encoder = DepthEncoder(features_dim)

        # LSTM: input = features(64) + state(7) = 71
        # No state_fc! Raw state goes directly into LSTM.
        self.lstm = nn.LSTM(
            input_size=features_dim + states_dim,  # 64 + 7 = 71
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # MLP extractor: input = lstm_out(256) + state(7) = 263
        mlp_input_size = lstm_hidden + states_dim  # 256 + 7 = 263

        # Actor (policy network): 263 -> [256, 256] -> 4
        self.actor = nn.Sequential(
            nn.Linear(mlp_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.action_net = nn.Linear(256, act_dim)

        # Critic (value network): 263 -> [512, 512] -> 1
        self.critic = nn.Sequential(
            nn.Linear(mlp_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self._lstm_hidden_shape = (1, 1, lstm_hidden)

    def forward(self, image, state, lstm_hidden=None):
        """
        Forward pass through the full network.

        Args:
            image: (batch, seq, 1, 256, 256) or (batch*seq, 1, 256, 256)
            state: (batch, seq, 7) - raw goal-oriented state
            lstm_hidden: ((num_layers, batch, hidden), ...) or None

        Returns:
            action_mean: (batch, seq, act_dim)
            value: (batch, seq, 1)
            new_hidden: ((num_layers, batch, hidden), ...)
        """
        B, T = image.shape[:2]

        # 1. Encode images: reshape to (B*T, 1, 256, 256)
        img_flat = image.reshape(B * T, 1, 256, 256)
        features = self.encoder(img_flat)  # (B*T, 64)

        # 2. Raw state (NO fc encoding, matching MAVRL)
        state_flat = state.reshape(B * T, self.states_dim)  # (B*T, 7)

        # 3. LSTM input: [features, state] = 64 + 7 = 71
        lstm_in = torch.cat([features, state_flat], dim=-1)  # (B*T, 71)
        lstm_in = lstm_in.reshape(B, T, -1)

        if lstm_hidden is None:
            lstm_out, new_hidden = self.lstm(lstm_in)
        else:
            lstm_out, new_hidden = self.lstm(lstm_in, lstm_hidden)

        # 4. MLP input: [lstm_out, state] = 256 + 7 = 263
        state_seq = state.reshape(B, T, self.states_dim)
        mlp_in = torch.cat([lstm_out, state_seq], dim=-1)  # (B, T, 263)

        # 5. Actor
        actor_features = self.actor(mlp_in)
        action_mean = self.action_net(actor_features)  # (B, T, 4)

        # 6. Critic
        value = self.critic(mlp_in)  # (B, T, 1)

        return action_mean, value, new_hidden

    def get_initial_hidden(self, batch_size=1, device='cpu'):
        hidden = (
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
        )
        return hidden

    def predict(self, image, state, lstm_hidden=None, deterministic=True):
        """
        Inference: returns action (after Tanh) and new hidden state.
        Matching MAVRL: action = Tanh(action_mean)
        """
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if state.dim() == 3 and state.shape[0] == 1:
                pass  # already batch dim
            elif state.dim() == 2:
                state = state.unsqueeze(0)

            mean, value, new_hidden = self.forward(image, state, lstm_hidden)

            # Take last timestep
            mean = mean[:, -1]  # (batch, 4)

            if deterministic:
                action = torch.tanh(mean)
            else:
                # During training, add exploration noise
                action = torch.tanh(mean + torch.randn_like(mean) * 0.1)

            return action.cpu().numpy(), new_hidden

    def evaluate_actions(self, image, state, actions, lstm_hidden=None):
        """
        Evaluate actions for PPO update.
        Returns value, log_prob, entropy.
        """
        mean, value, _ = self.forward(image, state, lstm_hidden)

        mean_flat = mean.reshape(-1, self.act_dim)
        value_flat = value.reshape(-1)

        # Gaussian distribution for PPO (exploration during training)
        # log_std is fixed at -0.5 (MAVRL default)
        log_std = torch.full((self.act_dim,), -0.5, device=mean.device)
        std = log_std.exp()
        dist = Normal(mean_flat, std)

        actions_flat = actions.reshape(-1, self.act_dim)
        log_prob = dist.log_prob(actions_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return value_flat, log_prob, entropy

    # --- MAVRL-style split methods for RecurrentPPO ---

    def forward_rnn(self, image, state, lstm_hidden):
        """
        Step 1: Run encoder + LSTM. Returns latent vectors for buffer storage.
        Called ONCE during rollout collection.

        Returns:
            latent_pi: (batch, lstm_hidden + state_dim) = (batch, 263)
            latent_vf: (batch, lstm_hidden + state_dim) = (batch, 263)
            new_hidden, new_cell
        """
        B, T = image.shape[:2]

        # Encode images
        img_flat = image.reshape(B * T, 1, 256, 256)
        features = self.encoder(img_flat)  # (B*T, 64)

        # Raw state
        state_flat = state.reshape(B * T, self.states_dim)  # (B*T, 7)

        # LSTM input: [features, state] = 71
        lstm_in = torch.cat([features, state_flat], dim=-1)
        lstm_in = lstm_in.reshape(B, T, -1)

        lstm_out, new_hidden = self.lstm(lstm_in, lstm_hidden)

        # Latent = [lstm_out, state] = 263
        state_seq = state.reshape(B, T, self.states_dim)
        latent_pi = torch.cat([lstm_out, state_seq], dim=-1)  # (B, T, 263)
        latent_vf = latent_pi  # Shared (like MAVRL shared_lstm=True)

        # Take last timestep for buffer storage
        latent_pi = latent_pi[:, -1]  # (B, 263)
        latent_vf = latent_vf[:, -1]  # (B, 263)

        return latent_pi, latent_vf, new_hidden[0], new_hidden[1]

    def forward_from_latent(self, latent_pi, latent_vf):
        """
        Step 2: Run MLP on pre-computed latents. Returns action and value.
        Called after forward_rnn during rollout.

        Args:
            latent_pi: (batch, 263) or (batch, 1, 263)
            latent_vf: (batch, 263) or (batch, 1, 263)
        """
        if latent_pi.dim() == 3:
            latent_pi = latent_pi[:, -1]
            latent_vf = latent_vf[:, -1]

        # Actor
        actor_features = self.actor(latent_pi)
        action_mean = self.action_net(actor_features)  # (batch, 4)

        # Critic
        value = self.critic(latent_vf)  # (batch, 1)

        return action_mean, value

    def evaluate_actions_from_latent(self, latent_pi, latent_vf, actions):
        """
        Evaluate actions for PPO update using PRE-COMPUTED latents.
        NO encoder/LSTM forward pass!
        """
        # Actor
        actor_features = self.actor(latent_pi)
        action_mean = self.action_net(actor_features)

        # Critic
        value = self.critic(latent_vf).squeeze(-1)

        # Distribution (log_std fixed at -0.5, MAVRL default)
        log_std = torch.full((self.act_dim,), -0.5, device=latent_pi.device)
        dist = Normal(action_mean, log_std.exp())

        actions_flat = actions.reshape(-1, self.act_dim)
        log_prob = dist.log_prob(actions_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return value, log_prob, entropy


class MultiInputLstmPolicy(nn.Module):
    """
    Wrapper for SB3 compatibility with Dict observation space.
    Matches MAVRL MultiInputLstmPolicy.
    """

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

        mean, value, _ = self.policy.forward(image, state)

        mean = mean[:, -1]
        value = value[:, -1]

        if deterministic:
            action = torch.tanh(mean)
        else:
            action = torch.tanh(mean + torch.randn_like(mean) * 0.1)

        return action, value, None

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

    # --- MAVRL-style methods for RecurrentPPO ---

    def forward_rnn(self, image, state, lstm_hidden):
        """Run encoder + LSTM, return latent vectors."""
        return self.policy.forward_rnn(image, state, lstm_hidden)

    def forward_from_latent(self, latent_pi, latent_vf):
        """Run MLP on pre-computed latents."""
        return self.policy.forward_from_latent(latent_pi, latent_vf)

    def evaluate_actions_from_latent(self, latent_pi, latent_vf, actions):
        """PPO update on pre-computed latents."""
        return self.policy.evaluate_actions_from_latent(latent_pi, latent_vf, actions)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
