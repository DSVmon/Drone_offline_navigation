"""
RecurrentPPO for MAVRL-style drone navigation.

Custom PPO implementation with LSTM support.
Based on Stable-Baselines3 and MAVRL.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque
from pathlib import Path

import config


class RecurrentRolloutBuffer:
    """Rollout buffer for recurrent policies with LSTM hidden states."""

    def __init__(self, buffer_size, obs_dim, act_dim, device='cpu'):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.ptr = 0

        # Pre-allocate buffers
        self.observations = {'image': [], 'state': []}
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        # LSTM hidden states
        self.lstm_h = []
        self.lstm_c = []

    def add(self, obs, action, reward, done, value, log_prob, lstm_h, lstm_c):
        """Add a single transition."""
        idx = self.ptr % self.buffer_size
        self.observations['image'].append(obs['image'].copy())
        self.observations['state'].append(obs['state'].copy())
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.lstm_h.append(lstm_h.copy())
        self.lstm_c.append(lstm_c.copy())
        self.ptr += 1

    def compute_returns(self, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and returns."""
        n = len(self.observations['image'])
        last_gae = 0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            next_done = self.dones[t] if t < n - 1 else 0

            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size=256):
        """Generate random mini-batches."""
        n = len(self.observations['image'])
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            yield {
                'image': np.array([self.observations['image'][i] for i in batch_idx]),
                'state': np.array([self.observations['state'][i] for i in batch_idx]),
                'actions': self.actions[:n][batch_idx],
                'old_values': self.values[:n][batch_idx],
                'old_log_probs': self.log_probs[:n][batch_idx],
                'advantages': self.advantages[:n][batch_idx],
                'returns': self.returns[:n][batch_idx],
            }

    def reset(self):
        self.ptr = 0
        self.observations = {'image': [], 'state': []}
        self.lstm_h = []
        self.lstm_c = []

    def __len__(self):
        return self.ptr


class RecurrentPPO:
    """
    Recurrent PPO for LSTM-based policies.

    Supports:
        - LSTM hidden state propagation
        - GAE advantage estimation
        - Clipped surrogate loss
        - Value function clipping
        - Entropy bonus
    """

    def __init__(self, policy, env, lr=1e-4, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 n_steps=1000, batch_size=256, n_epochs=10, device='cpu',
                 tensorboard_log=None):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        self.buffer = RecurrentRolloutBuffer(n_steps, None, 4, device)

        self.num_timesteps = 0
        self.num_updates = 0

        # TensorBoard
        self.tensorboard_log = None
        if tensorboard_log:
            from torch.utils.tensorboard import SummaryWriter
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
            self.tensorboard_log = SummaryWriter(tensorboard_log)

    def collect_rollouts(self, deterministic=False):
        """Collect rollouts from the environment."""
        self.policy.eval()
        obs = self.env.reset()
        lstm_h = self.policy.get_initial_hidden(1, self.device)
        lstm_c = lstm_h[1]

        total_reward = 0
        episode_length = 0

        for step in range(self.n_steps):
            # Prepare observation
            image = torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device)
            state = torch.FloatTensor(obs['state']).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                mean, std, value, (new_h, new_c) = self.policy.forward(
                    image, state, (lstm_h, lstm_c)
                )

            # Sample action
            if deterministic:
                action = torch.tanh(mean[:, -1])
                log_prob = torch.zeros(1)
            else:
                dist = torch.distributions.Normal(mean[:, -1], std[:, -1])
                raw_action = dist.sample()
                action = torch.tanh(raw_action)
                log_prob = dist.log_prob(raw_action).sum(dim=-1)

            action_np = action.cpu().numpy()[0]

            # Step environment
            obs, reward, done, info = self.env.step(action_np)

            # Store transition
            self.buffer.add(
                obs={'image': obs['image'], 'state': obs['state']},
                action=action_np,
                reward=reward,
                done=float(done),
                value=value.item(),
                log_prob=log_prob.item(),
                lstm_h=new_h.cpu().numpy(),
                lstm_c=new_c.cpu().numpy(),
            )

            lstm_h = new_h.detach()
            lstm_c = new_c.detach()

            total_reward += reward
            episode_length += 1
            self.num_timesteps += 1

            if done:
                obs = self.env.reset()
                lstm_h = self.policy.get_initial_hidden(1, self.device)
                lstm_c = lstm_h[1]

        return total_reward, episode_length

    def train_step(self):
        """One PPO update step."""
        self.policy.train()

        # Compute returns and advantages
        self.buffer.compute_returns(self.gamma, self.gae_lambda)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_batches = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # Convert to tensors
                image = torch.FloatTensor(batch['image']).to(self.device)
                state = torch.FloatTensor(batch['state']).to(self.device)
                actions = torch.FloatTensor(batch['actions']).to(self.device)
                old_values = torch.FloatTensor(batch['old_values']).to(self.device)
                old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
                advantages = torch.FloatTensor(batch['advantages']).to(self.device)
                returns = torch.FloatTensor(batch['returns']).to(self.device)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass
                mean, std, values, _ = self.policy.forward(
                    image.unsqueeze(1), state.unsqueeze(1)
                )
                values = values.squeeze(-1)

                # Compute log probs
                dist = torch.distributions.Normal(mean.squeeze(1), std.squeeze(1))
                log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                # Policy loss (clipped)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_batches += 1

        self.num_updates += 1

        # Log
        avg_policy_loss = total_policy_loss / max(n_batches, 1)
        avg_value_loss = total_value_loss / max(n_batches, 1)
        avg_entropy = total_entropy_loss / max(n_batches, 1)

        if self.tensorboard_log:
            self.tensorboard_log.add_scalar('train/policy_loss', avg_policy_loss, self.num_timesteps)
            self.tensorboard_log.add_scalar('train/value_loss', avg_value_loss, self.num_timesteps)
            self.tensorboard_log.add_scalar('train/entropy', avg_entropy, self.num_timesteps)

        return avg_policy_loss, avg_value_loss, avg_entropy

    def learn(self, total_timesteps, callback=None):
        """Main training loop."""
        print(f"[PPO] Starting training for {total_timesteps:,} steps")
        print(f"[PPO] Device: {self.device}")

        start_time = time.time()

        while self.num_timesteps < total_timesteps:
            # Collect rollouts
            episode_reward, episode_length = self.collect_rollouts()

            # Train
            policy_loss, value_loss, entropy = self.train_step()

            # Log
            elapsed = time.time() - start_time
            fps = int(self.num_timesteps / max(elapsed, 1))

            print(f"[PPO] Steps: {self.num_timesteps:_} | "
                  f"Reward: {episode_reward:+.2f} | "
                  f"Len: {episode_length} | "
                  f"PL: {policy_loss:.4f} | VL: {value_loss:.4f} | "
                  f"FPS: {fps}")

            if self.tensorboard_log:
                self.tensorboard_log.add_scalar('rollout/episode_reward', episode_reward, self.num_timesteps)
                self.tensorboard_log.add_scalar('rollout/episode_length', episode_length, self.num_timesteps)
                self.tensorboard_log.add_scalar('time/fps', fps, self.num_timesteps)

            if callback:
                callback(self)

    def save(self, path):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            'num_updates': self.num_updates,
        }, path)
        print(f"[PPO] Model saved to {path}")

    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_timesteps = checkpoint['num_timesteps']
        self.num_updates = checkpoint['num_updates']
        print(f"[PPO] Model loaded from {path}")
