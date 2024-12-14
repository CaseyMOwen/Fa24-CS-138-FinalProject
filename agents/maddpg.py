"""
This script demonstrates an enhanced version of the MADDPG integration into the CityLearn environment.
It incorporates:
- Normalizing observations.
- Using Ornstein-Uhlenbeck (OU) noise for exploration.
- Implementing a warm-up period before training starts.
- Adjusting training duration and logging intermediate metrics.
- Providing reward scaling and improved logging for diagnostics.

Assumptions & Notes:
--------------------
1. Requires CityLearn environment and dataset:
   from citylearn.citylearn import CityLearnEnv

2. We assume a scenario with two buildings (agents). The code sets building_ids = [0, 1]. Adjust as needed.

3. Observations are assumed to be [hour, month, indoor_dry_bulb_temperature]. This code normalizes them:
   - hour: normalized by dividing by 24.0
   - month: normalized by dividing by 12.0
   - indoor_dry_bulb_temperature: assumed to be in roughly [0°C, 40°C] for demonstration, and normalized to [0,1] by dividing by 40.
   Adjust this normalization as needed for your specific scenario and data ranges.

4. Rewards may be too sparse or not aligned with battery usage goals. Consider improving the reward function within the environment or scaling rewards if not learning effectively.
   In this code, we add a reward scaling factor.

5. A warm-up period is introduced (warmup_steps = 1000), during which experience is collected but no training occurs.

6. Ornstein-Uhlenbeck (OU) noise is added for better exploration. Hyperparameters for OU noise can be tuned.

7. The code logs episode rewards and prints intermediate Q-values and actions periodically for debugging.

8. Increase the number of episodes and consider running for longer to allow the policy to converge.

This code is a template. Adjust hyperparameters, normalization ranges, reward structure, and exploration parameters as needed.
"""

import os
import csv
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from citylearn.citylearn import CityLearnEnv

# ==========================
# Setup Logging
# ==========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# Replay Buffer
# ==========================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, num_agents, buffer_size=int(2e6), batch_size=512, device='cpu'):
        """
        A ReplayBuffer for storing experience tuples (o, a, r, o', d).
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self.obs_buf = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((buffer_size, num_agents, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.done_buf = np.zeros((buffer_size, 1), dtype=np.float32)

        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.next_obs_buf[idx] = next_obs
        self.done_buf[idx] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample_batch(self):
        if self.size < self.batch_size:
            return None
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(
            obs=torch.FloatTensor(self.obs_buf[idxs]).to(self.device),
            act=torch.FloatTensor(self.act_buf[idxs]).to(self.device),
            rew=torch.FloatTensor(self.rew_buf[idxs]).to(self.device),
            next_obs=torch.FloatTensor(self.next_obs_buf[idxs]).to(self.device),
            done=torch.FloatTensor(self.done_buf[idxs]).to(self.device),
        )
        return batch

# ==========================
# Neural Networks
# ==========================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256], activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        layers = []
        prev_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(activation())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256,256], device='cpu'):
        super().__init__()
        self.device = device
        self.net = MLP(obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, output_activation=nn.Tanh).to(device)

    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_act_dim, hidden_sizes=[256, 256], device='cpu'):
        super().__init__()
        self.device = device
        self.net = MLP(total_obs_dim + total_act_dim, 1, hidden_sizes, activation=nn.ReLU, output_activation=None).to(device)

    def forward(self, obs_all, act_all):
        x = torch.cat([obs_all, act_all], dim=-1)
        return self.net(x)

# ==========================
# Ornstein-Uhlenbeck Noise
# ==========================
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x_prev = np.zeros(self.size)

    def reset(self):
        self.x_prev = np.zeros(self.size)

    def sample(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.randn(self.size)
        self.x_prev = x
        return x

# ==========================
# MADDPG Agent
# ==========================
class MADDPGAgent:
    def __init__(self, num_agents, obs_dim, act_dim, gamma=0.99, lr=1e-3, tau=0.005, device='cpu'):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.device = device

        # Actors and Critics
        self.actors = [Actor(obs_dim, act_dim, device=device) for _ in range(num_agents)]
        self.actor_targets = deepcopy(self.actors)
        self.critics = [Critic(num_agents*obs_dim, num_agents*act_dim, device=device) for _ in range(num_agents)]
        self.critic_targets = deepcopy(self.critics)

        self.actor_optimizers = [optim.Adam(a.parameters(), lr=self.lr) for a in self.actors]
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=self.lr) for c in self.critics]

        # OU Noise per agent
        self.noises = [OUNoise(act_dim) for _ in range(num_agents)]

    def select_action(self, obs, add_noise=True):
        actions = []
        for i, actor in enumerate(self.actors):
            obs_i = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
            act_i = actor(obs_i).cpu().detach().numpy()[0]
            if add_noise:
                # Use OU noise for better exploration
                noise = self.noises[i].sample()
                act_i += noise
            act_i = np.clip(act_i, -1, 1)
            actions.append(act_i)
        return np.array(actions)

    def update(self, replay_buffer):
        batch = replay_buffer.sample_batch()
        if batch is None:
            return
        obs, act, rew, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        # Flatten
        obs_all = obs.reshape(obs.shape[0], -1)
        act_all = act.reshape(act.shape[0], -1)
        next_obs_all = next_obs.reshape(next_obs.shape[0], -1)

        with torch.no_grad():
            next_act = []
            for i, actor_targ in enumerate(self.actor_targets):
                next_act_i = actor_targ(next_obs[:, i, :])
                next_act.append(next_act_i)
            next_act_all = torch.cat(next_act, dim=-1)

        # Update critics
        for i in range(self.num_agents):
            q_target = self.critic_targets[i](next_obs_all, next_act_all)
            y = rew[:, i].unsqueeze(-1) + self.gamma * (1 - done)*q_target
            q_val = self.critics[i](obs_all, act_all)
            critic_loss = nn.MSELoss()(q_val, y)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # Update actors
        for i in range(self.num_agents):
            act_i_list = []
            for j, actor in enumerate(self.actors):
                if j == i:
                    act_i_list.append(actor(obs[:, j, :]))
                else:
                    act_i_list.append(self.actors[j](obs[:, j, :]).detach())
            act_i_all = torch.cat(act_i_list, dim=-1)
            actor_loss = -self.critics[i](obs_all, act_i_all).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Soft update
        self._soft_update()

    def _soft_update(self):
        for i in range(self.num_agents):
            for param, target_param in zip(self.critics[i].parameters(), self.critic_targets[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau)*target_param.data)
            for param, target_param in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau)*target_param.data)

    def reset_noise(self):
        for noise in self.noises:
            noise.reset()

# ==========================
# Utility Functions
# ==========================

def normalize_observation(obs):
    """
    Normalize the given observation vector [hour, month, indoor_temp].
    Adjust ranges as needed for your environment.
    hour in [0,23], month in [1,12], indoor_temp in [0°C,40°C] (example).
    """
    hour = obs[0] / 24.0
    month = obs[1] / 12.0
    # Assume indoor_temp roughly in [0,40]
    indoor_temp = obs[2] / 40.0
    return np.array([hour, month, indoor_temp], dtype=np.float32)

# ==========================
# Main Training
# ==========================
if __name__ == "__main__":
    DATASET_NAME = 'citylearn_challenge_2022_phase_all'
    env = CityLearnEnv(schema=DATASET_NAME)

    building_ids = [0, 1]
    env.building_ids = building_ids
    obs = env.reset()

    # Normalize initial observation
    obs = [normalize_observation(o) for o in obs]

    num_agents = len(building_ids)
    obs_dim = len(obs[0])
    act_dim = env.action_space[0].shape[0]

    # Instantiate MADDPG
    agent = MADDPGAgent(num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim, device='cpu')

    # Replay Buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, num_agents=num_agents, device='cpu')

    num_episodes = 100             # increased from 10 to allow more learning
    steps_per_episode = 24 * 7     # 168 steps/episode for 7 days hourly
    warmup_steps = 1000            # warm-up period before training
    reward_scale = 0.01            # scale rewards if too large/small
    log_file = "maddpg_training_log.csv"

    steps_collected = 0

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total_Reward"])
        for ep in range(num_episodes):
            obs = env.reset()
            obs = [normalize_observation(o) for o in obs]
            agent.reset_noise()
            total_reward = 0.0

            for t in range(steps_per_episode):
                action = agent.select_action(obs, add_noise=True)
                next_obs, rewards, done, info = env.step(action)

                # If scalar reward, broadcast to all agents
                if np.isscalar(rewards):
                    rewards = np.array([rewards]*num_agents)

                # Normalize next_obs
                next_obs = [normalize_observation(o) for o in next_obs]

                # Scale rewards
                rewards = rewards * reward_scale

                replay_buffer.store(np.array(obs), np.array(action), rewards, np.array(next_obs), np.array([done]))
                steps_collected += 1

                obs = next_obs
                total_reward += sum(rewards)

                # Train only after warmup
                if steps_collected > warmup_steps:
                    agent.update(replay_buffer)

                if done:
                    break

            # Logging per episode
            writer.writerow([ep, total_reward])
            logger.info(f"Episode {ep}, Total Reward: {total_reward}")

            # Debug logging: check actions and Q-values periodically
            if ep % 10 == 0:
                # Sample Q-value for debugging
                test_batch = replay_buffer.sample_batch()
                if test_batch:
                    test_obs = test_batch['obs'].reshape(-1, num_agents*obs_dim)
                    test_act = test_batch['act'].reshape(-1, num_agents*act_dim)
                    q_values = agent.critics[0](test_obs, test_act)
                    logger.info(f"Episode {ep}: mean Q-value (agent0) = {q_values.mean().item():.3f}")

                logger.info(f"Episode {ep}: Example actions last step = {action}, reward this episode = {total_reward}")

    logger.info("Training complete! Check maddpg_training_log.csv for results.")
