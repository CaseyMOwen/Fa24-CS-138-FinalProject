"""
Assumptions & Notes:
--------------------
1. In this example, we are controlling two radomly selected buildings (agents) assuming the environment 
   provides appropriate observations and action spaces for these buildings. We are using a simplified 
   observation space: ACTIVE_OBSERVATIONS = ['hour', 'month', 'indoor_dry_bulb_temperature']. 

2. The action space is continuous and expected to be in the range [-1, 1]. The exact dimension 
   depends on the number and type of controllable devices. Here we assume that each agent (building) 
   has at least one continuous action dimension.

3. MADDPG involves:
   - A decentralized actor for each agent (outputting continuous actions).
   - A centralized critic for each agent that conditions on all agents' observations and actions.

4. Hyperparameters have defauly values assigned while we've provided comments in the code that suggest alternative choices.
   For example, you can modify the learning rate, network architecture, or tau if training is unstable.

5. This script sets up the environment, trains for a number of episodes, and logs episode-level 
   rewards to a CSV file. 

6. Custom logging of additional metrics or using frameworks like TensorBoard can be easily integrated 
   if desired. Here we show basic CSV logging of total episode rewards.

7. To run:
   `maddpg.py`

   Ensure that:
   - The dataset schema ('citylearn_challenge_2022_phase_all') is available.
   - CityLearn and its dependencies are installed.
   - You have write access to create "maddpg_training_log.csv" in the current directory.
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
# Replay Buffer for Experience Storage
# ==========================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, num_agents, buffer_size=int(1e6), batch_size=256, device='cpu'):
        """
        A simple replay buffer that stores tuples of (obs, act, rew, next_obs, done).

        Parameters:
        - obs_dim: Dimension of single-agent observation vector.
        - act_dim: Dimension of single-agent action vector.
        - num_agents: Number of agents in the environment.
        - buffer_size: Maximum number of transitions to store.
        - batch_size: Number of samples per training batch.
        - device: 'cpu' or 'cuda', for PyTorch tensor placement.
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
        """
        Store a single transition in the replay buffer.
        """
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.next_obs_buf[idx] = next_obs
        self.done_buf[idx] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample_batch(self):
        """
        Sample a random batch of transitions for training.
        Returns a dictionary of PyTorch tensors on the designated device.
        """
        if self.size < self.batch_size:
            return None  # Not enough samples yet

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
# Neural Network Utilities
# ==========================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256], activation=nn.ReLU, output_activation=nn.Identity):
        """
        A simple Multi-Layer Perceptron.
        Can modify hidden_sizes, activation as needed.
        """
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
        """
        Actor network for deterministic policy, outputs continuous actions in range [-1, 1].
        """
        super().__init__()
        self.device = device
        # Output activation: Tanh to ensure actions are in [-1, 1]
        self.net = MLP(obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, output_activation=nn.Tanh).to(device)

    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_act_dim, hidden_sizes=[256, 256], device='cpu'):
        """
        Centralized critic that takes as input all agents' observations and actions.
        """
        super().__init__()
        self.device = device
        # Single scalar Q-value output
        self.net = MLP(total_obs_dim + total_act_dim, 1, hidden_sizes, activation=nn.ReLU, output_activation=None).to(device)

    def forward(self, obs_all, act_all):
        """
        obs_all: (batch, num_agents*obs_dim)
        act_all: (batch, num_agents*act_dim)
        """
        x = torch.cat([obs_all, act_all], dim=-1)
        return self.net(x)

# ==========================
# MADDPG Agent Class
# ==========================
class MADDPGAgent:
    def __init__(self, num_agents, obs_dim, act_dim, gamma=0.99, lr=1e-3, tau=0.005, device='cpu'):
        """
        Initialize MADDPG for a set of agents.

        Parameters:
        - num_agents: Number of agents (e.g., number of buildings).
        - obs_dim: Dimensionality of each agent's observation.
        - act_dim: Dimensionality of each agent's action.
        - gamma: Discount factor.
        - lr: Learning rate for both actors and critics.
        - tau: Soft update parameter for target networks.
        - device: 'cpu' or 'cuda'.
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.device = device

        # Create actor and critic networks for each agent
        self.actors = [Actor(obs_dim, act_dim, device=device) for _ in range(num_agents)]
        self.actor_targets = deepcopy(self.actors)
        self.critics = [Critic(num_agents*obs_dim, num_agents*act_dim, device=device) for _ in range(num_agents)]
        self.critic_targets = deepcopy(self.critics)

        self.actor_optimizers = [optim.Adam(a.parameters(), lr=self.lr) for a in self.actors]
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=self.lr) for c in self.critics]

        # Exploration noise settings
        self.noise_std = 0.1

    def select_action(self, obs):
        """
        Select action for each agent based on current actor and add exploration noise.
        obs: list or array of shape [num_agents, obs_dim]
        """
        actions = []
        for i, actor in enumerate(self.actors):
            obs_i = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
            act_i = actor(obs_i).cpu().detach().numpy()[0]
            # Add Gaussian noise for exploration
            act_i += np.random.normal(0, self.noise_std, size=self.act_dim)
            act_i = np.clip(act_i, -1, 1)
            actions.append(act_i)
        return np.array(actions)

    def update(self, replay_buffer):
        """
        Update actors and critics using one batch sample from the replay buffer.
        Performs a soft update of target networks afterward.
        """
        batch = replay_buffer.sample_batch()
        if batch is None:
            return  # Not enough data yet

        obs, act, rew, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']
        # Flatten along agent dimension
        obs_all = obs.reshape(obs.shape[0], -1)
        act_all = act.reshape(act.shape[0], -1)
        next_obs_all = next_obs.reshape(next_obs.shape[0], -1)

        # Compute next actions from target actors
        with torch.no_grad():
            next_act = []
            for i, actor_targ in enumerate(self.actor_targets):
                next_act_i = actor_targ(next_obs[:, i, :])
                next_act.append(next_act_i)
            next_act_all = torch.cat(next_act, dim=-1)

        # Update critics
        for i in range(self.num_agents):
            q_target = self.critic_targets[i](next_obs_all, next_act_all)
            y = rew[:, i].unsqueeze(-1) + self.gamma * (1 - done) * q_target
            q_val = self.critics[i](obs_all, act_all)
            critic_loss = nn.MSELoss()(q_val, y)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # Update actors
        for i in range(self.num_agents):
            # For the i-th actor, we need to compute the actor loss
            act_i_list = []
            for j, actor in enumerate(self.actors):
                if j == i:
                    act_i_list.append(actor(obs[:, j, :]))
                else:
                    # Other agents' actions no gradient wrt this actor
                    act_i_list.append(self.actors[j](obs[:, j, :]).detach())
            act_i_all = torch.cat(act_i_list, dim=-1)
            actor_loss = -self.critics[i](obs_all, act_i_all).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Soft update targets
        self._soft_update()

    def _soft_update(self):
        """
        Soft update target networks using (tau * local + (1 - tau) * target).
        """
        for i in range(self.num_agents):
            for param, target_param in zip(self.critics[i].parameters(), self.critic_targets[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ==========================
# Main Training Procedure
# ==========================
if __name__ == "__main__":
    # Define the dataset name or path
    DATASET_NAME = 'citylearn_challenge_2022_phase_all'

    # Create the CityLearn environment
    env = CityLearnEnv(schema=DATASET_NAME)

    # Randomly select 2 buildings or define them explicitly
    building_ids = [0, 1]
    env.building_ids = building_ids  # This may differ based on how you've set up your code

    obs = env.reset()
    num_agents = len(building_ids)

    # We assume obs is something like a list of [obs_for_building_0, obs_for_building_1]
    # Each obs_for_building_i is a vector of dimension obs_dim.
    obs_dim = len(obs[0])
    # Action dimension depends on environment's action space
    # Assume each building has identical action dimension
    act_dim = env.action_space[0].shape[0]

    # Initialize MADDPG
    agent = MADDPGAgent(num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim, device='cpu')

    # Create a replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, num_agents=num_agents, device='cpu')

    # Training parameters
    num_episodes = 10            # Increase for more training
    steps_per_episode = 24 * 7   # Assuming 7 days at hourly steps = 168 steps/episode
    log_file = "maddpg_training_log.csv"

    # Open CSV for logging
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total_Reward"])  # Header for logging

        for ep in range(num_episodes):
            obs = env.reset()
            total_reward = 0.0
            for t in range(steps_per_episode):
                # Select actions with exploration
                action = agent.select_action(obs)

                # Step environment
                next_obs, rewards, done, info = env.step(action)

                # If rewards is a scalar or single value, broadcast to all agents
                if np.isscalar(rewards):
                    rewards = np.array([rewards]*num_agents)

                # Store the transition
                replay_buffer.store(np.array(obs), np.array(action), rewards, np.array(next_obs), np.array([done]))

                # Update the agent (assuming enough samples)
                agent.update(replay_buffer)

                obs = next_obs
                total_reward += sum(rewards)

                if done:
                    break

            logger.info(f"Episode {ep}, Total Reward: {total_reward}")
            writer.writerow([ep, total_reward])

    logger.info("Training complete!")
    logger.info(f"Training log saved to {log_file}")

    # After training, one can evaluate the policy on unseen subsets by resetting the environment 
    # with different conditions and disabling exploration noise (set agent.noise_std = 0).
