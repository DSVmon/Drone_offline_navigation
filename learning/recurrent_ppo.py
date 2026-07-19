"""
RecurrentPPO for MAVRL-style drone navigation.

Key MAVRL pattern:
    1. collect_rollouts: forward_rnn(obs) -> latent_pi, latent_vf
       Store latents in buffer (NOT raw observations)
    2. train_step: evaluate_actions(latent_pi, latent_vf, actions)
       NO re-computation of encoder/LSTM during PPO update

Reference: github.com/tudelft/mavrl/mav_baselines/torch/recurrent_ppo/ppo_recurrent.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

import config


class RecurrentRolloutBuffer:
    """
    Rollout buffer that stores pre-computed LSTM latents.

    MAVRL stores latent_pi/latent_vf (LSTM output) instead of raw observations.
    This means PPO update does NOT re-run the encoder/LSTM.
    """

    def __init__(self, buffer_size, act_dim, lstm_dim=256, state_dim=7, device='cpu'):
        self.buffer_size = buffer_size
        self.act_dim = act_dim
        self.lstm_dim = lstm_dim
        self.state_dim = state_dim
        self.device = device
        self.ptr = 0

        # Latent vectors (pre-computed by LSTM)
        self.latent_pi = np.zeros((buffer_size, lstm_dim + state_dim), dtype=np.float32)
        self.latent_vf = np.zeros((buffer_size, lstm_dim + state_dim), dtype=np.float32)

        # Standard PPO data
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        # LSTM hidden states (for sequence boundary detection)
        self.lstm_h = []
        self.lstm_c = []

    def add(self, latent_pi, latent_vf, action, reward, done, value, log_prob, lstm_h, lstm_c):
        """Add a single transition with pre-computed latents."""
        idx = self.ptr % self.buffer_size
        self.latent_pi[idx] = latent_pi
        self.latent_vf[idx] = latent_vf
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
        n = self.ptr
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

    def get_batches(self, batch_size=4000):
        """
        Generate random mini-batches of PRE-COMPUTED latents.
        No raw observations are returned.
        """
        n = self.ptr
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            yield {
                'latent_pi': self.latent_pi[batch_idx],
                'latent_vf': self.latent_vf[batch_idx],
                'actions': self.actions[batch_idx],
                'old_values': self.values[batch_idx],
                'old_log_probs': self.log_probs[batch_idx],
                'advantages': self.advantages[batch_idx],
                'returns': self.returns[batch_idx],
            }

    def reset(self):
        self.ptr = 0
        self.lstm_h = []
        self.lstm_c = []

    def __len__(self):
        return self.ptr


class RecurrentPPO:
    """
    Recurrent PPO for LSTM-based policies.
    Matches MAVRL RecurrentPPO data flow exactly.

    Key insight: LSTM is called ONCE during rollout collection.
    PPO update uses pre-computed latent vectors.
    """

    def __init__(self, policy, env, lr=1e-4, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.2, ent_coef=0.0, vf_coef=0.2, max_grad_norm=0.5,
                 n_steps=1000, batch_size=4000, n_epochs=10, device='cpu',
                 tensorboard_log=None, lr_schedule=None, eval_env=None,
                 eval_freq=200_000, eval_episodes=3):
        self.policy = policy
        self.env = env
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
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
        self.lr_schedule = lr_schedule

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        lstm_dim = config.LSTM_HIDDEN_SIZE
        state_dim = config.STATE_DIM
        self.buffer = RecurrentRolloutBuffer(
            n_steps, config.ACTION_DIM, lstm_dim, state_dim, device
        )

        self.num_timesteps = 0
        self.num_updates = 0

        # TensorBoard
        self.tensorboard_log = None
        if tensorboard_log:
            from torch.utils.tensorboard import SummaryWriter
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
            self.tensorboard_log = SummaryWriter(tensorboard_log)

    def collect_rollouts(self, deterministic=False):
        """
        Collect rollouts from the environment.

        MAVRL flow:
            1. forward_rnn(obs) -> latent_pi, latent_vf (LSTM runs here)
            2. forward(latent_pi, latent_vf) -> actions, values
            3. Store latents in buffer
        """
        self.policy.eval()
        self.buffer.reset()
        reset_result = self.env.reset()
        # Handle both dict-only and (dict, info) return formats
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        lstm_h, lstm_c = self.policy.get_initial_hidden(1, self.device)

        total_reward = 0
        episode_length = 0
        episode_count = 0

        for step in range(self.n_steps):
            # Prepare observation tensors: (B=1, T=1, C, H, W) and (B=1, T=1, state_dim)
            image = torch.FloatTensor(obs['image']).reshape(1, 1, 1, config.DEPTH_HEIGHT, config.DEPTH_WIDTH).to(self.device)
            state = torch.FloatTensor(obs['state']).reshape(1, 1, config.STATE_DIM).to(self.device)

            # 1. Forward through encoder + LSTM (ONCE)
            with torch.no_grad():
                latent_pi, latent_vf, new_h, new_c = self.policy.forward_rnn(
                    image, state, (lstm_h, lstm_c)
                )

            # 2. Forward through MLP (actor/critic) on latent vectors
            with torch.no_grad():
                action_mean, value = self.policy.forward_from_latent(latent_pi, latent_vf)

            # 3. Sample action
            if deterministic:
                action = torch.tanh(action_mean)
                log_prob = torch.zeros(1, device=self.device)
            else:
                # Gaussian exploration (log_std fixed at -0.5, MAVRL default)
                log_std = torch.full((config.ACTION_DIM,), -0.5, device=self.device)
                dist = torch.distributions.Normal(action_mean, log_std.exp())
                raw_action = dist.sample()
                action = torch.tanh(raw_action)
                log_prob = dist.log_prob(raw_action).sum(dim=-1)

            action_np = action.cpu().numpy()[0]

            # 4. Step environment
            obs, reward, done, info = self.env.step(action_np)

            # 5. Store PRE-COMPUTED latents (not raw obs!)
            self.buffer.add(
                latent_pi=latent_pi.cpu().numpy()[0],
                latent_vf=latent_vf.cpu().numpy()[0],
                action=action_np,
                reward=reward,
                done=float(done),
                value=value.item(),
                log_prob=log_prob.item(),
                lstm_h=new_h.cpu().numpy(),
                lstm_c=new_c.cpu().numpy(),
            )

            # 6. Update LSTM hidden state, reset on episode boundary
            lstm_h = new_h.detach()
            lstm_c = new_c.detach()

            total_reward += reward
            episode_length += 1
            self.num_timesteps += 1

            if done:
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs = reset_result[0]
                else:
                    obs = reset_result
                lstm_h, lstm_c = self.policy.get_initial_hidden(1, self.device)
                episode_count += 1

        return total_reward, episode_length, episode_count

    def train_step(self):
        """
        One PPO update step using PRE-COMPUTED latents.
        NO encoder/LSTM forward pass here!
        """
        self.policy.train()

        # Compute returns and advantages
        self.buffer.compute_returns(self.gamma, self.gae_lambda)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_batches = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # Convert pre-computed latents to tensors
                latent_pi = torch.FloatTensor(batch['latent_pi']).to(self.device)
                latent_vf = torch.FloatTensor(batch['latent_vf']).to(self.device)
                actions = torch.FloatTensor(batch['actions']).to(self.device)
                old_values = torch.FloatTensor(batch['old_values']).to(self.device)
                old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
                advantages = torch.FloatTensor(batch['advantages']).to(self.device)
                returns = torch.FloatTensor(batch['returns']).to(self.device)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward through MLP ONLY (NO encoder/LSTM!)
                values, log_prob, entropy = self.policy.evaluate_actions_from_latent(
                    latent_pi, latent_vf, actions
                )

                # Policy loss (clipped)
                ratio = torch.exp(log_prob - old_log_probs)
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

    def learn(self, total_timesteps, callback=None, log_interval=10):
        """Main training loop."""
        print(f"[PPO] Starting training for {total_timesteps:,} steps")
        print(f"[PPO] Device: {self.device}")

        start_time = time.time()

        while self.num_timesteps < total_timesteps:
            # Collect rollouts
            episode_reward, episode_length, episode_count = self.collect_rollouts()

            # Train
            policy_loss, value_loss, entropy = self.train_step()

            # Log
            elapsed = time.time() - start_time
            fps = int(self.num_timesteps / max(elapsed, 1))

            if self.num_updates % log_interval == 0:
                print(f"[PPO] Steps: {self.num_timesteps:_} | "
                      f"Reward: {episode_reward:+.2f} | "
                      f"Episodes: {episode_count} | "
                      f"PL: {policy_loss:.4f} | VL: {value_loss:.4f} | "
                      f"FPS: {fps}")

            if self.tensorboard_log:
                self.tensorboard_log.add_scalar('rollout/episode_reward', episode_reward, self.num_timesteps)
                self.tensorboard_log.add_scalar('rollout/episode_length', episode_length, self.num_timesteps)
                self.tensorboard_log.add_scalar('rollout/episode_count', episode_count, self.num_timesteps)
                self.tensorboard_log.add_scalar('time/fps', fps, self.num_timesteps)

            if callback:
                callback(self)

            # Eval periodically (matching MAVRL eval pattern)
            if self.eval_env is not None and self.num_updates % (self.eval_freq // self.n_steps) == 0:
                self._run_eval()

    def _run_eval(self):
        """
        Run evaluation episodes on eval_env.
        MAVRL: eval() runs on medium/hard maps, saves trajectories.
        Ours: run N deterministic episodes, log success rate.
        """
        if self.eval_env is None:
            return

        print(f"\n[EVAL] Running {self.eval_episodes} evaluation episodes...")
        self.policy.eval()

        eval_rewards = []
        eval_lengths = []
        eval_successes = []

        for ep in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            lstm_h, lstm_c = self.policy.get_initial_hidden(1, self.device)

            total_reward = 0
            steps = 0
            done = False

            while not done and steps < 1000:
                image = torch.FloatTensor(obs['image']).reshape(1, 1, 1, config.DEPTH_HEIGHT, config.DEPTH_WIDTH).to(self.device)
                state = torch.FloatTensor(obs['state']).reshape(1, 1, config.STATE_DIM).to(self.device)

                with torch.no_grad():
                    latent_pi, latent_vf, new_h, new_c = self.policy.forward_rnn(
                        image, state, (lstm_h, lstm_c)
                    )
                    action_mean, _ = self.policy.forward_from_latent(latent_pi, latent_vf)
                    action = torch.tanh(action_mean).cpu().numpy()[0]

                obs, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                steps += 1
                lstm_h, lstm_c = new_h.detach(), new_c.detach()

            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
            eval_successes.append(1 if info.get('termination_reason') == 'completed_lap' else 0)

        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        success_rate = np.mean(eval_successes)

        print(f"[EVAL] Avg reward: {avg_reward:+.2f} | Avg length: {avg_length:.0f} | Success: {success_rate:.1%}")

        if self.tensorboard_log:
            self.tensorboard_log.add_scalar('eval/avg_reward', avg_reward, self.num_timesteps)
            self.tensorboard_log.add_scalar('eval/avg_length', avg_length, self.num_timesteps)
            self.tensorboard_log.add_scalar('eval/success_rate', success_rate, self.num_timesteps)

        self.policy.train()

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
