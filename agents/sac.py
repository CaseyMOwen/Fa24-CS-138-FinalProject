'''
Sources: 
https://github.com/ErickRosete/Robotics_Haptics/blob/c4232e5fbfdb6f6cce35755b2d3563facf8ee6b2/Soft_Actor_Critic/sac_agent.py#L114
https://www.citylearn.net/_modules/citylearn/agents/sac.html
'''

import math
from typing import Any, List, Tuple
import numpy as np
from .baseagent import Agent
from citylearn.citylearn import CityLearnEnv
import random
from sklearn.neural_network import MLPRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SAC(Agent):
    def __init__(
        self, env: CityLearnEnv, mini_batch_size:int=5, gamma:float=.99, tau:float=.005, entropy_coeff:float=0.2, **kwargs: Any,
    ):
        super().__init__(env, **kwargs)
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = entropy_coeff
        self.replay_buffer = [[] for _ in self.action_space]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.observation_dimension = [len(names) for names in self.observation_names]
        self.init_networks(200)

    def init_networks(self, hidden_dim:int):
        # Each variable is a list of several networks, one for each feature we are acting on (battery storage, thermal storage, etc.) 
        self.actors = [None for _ in self.action_space]
        # self.actor_targets = [None] * len(self.action_dimension)
        self.actor_optimizers = [None for _ in self.action_space]
        
        self.q1_nets = [None for _ in self.action_space]
        self.q1_targets = [None for _ in self.action_space]
        self.q1_optimizers = [None for _ in self.action_space]
        
        self.q2_nets = [None for _ in self.action_space]
        self.q2_targets = [None for _ in self.action_space]
        self.q2_optimizers = [None for _ in self.action_space]

        for i in range(len(self.action_space)):
            self.actors[i] = PolicyNetwork(self.observation_dimension[i], self.action_dimension[i], hidden_dim).to(self.device)
            self.actor_optimizers[i] = torch.optim.Adam(self.actors[i].parameters(), lr=3e-4)

            self.q1_nets[i] = QNetwork(self.observation_dimension[i], self.action_dimension[i], hidden_dim).to(self.device)
            self.q1_targets[i] = QNetwork(self.observation_dimension[i], self.action_dimension[i], hidden_dim).to(self.device)
            # Targets start with same parameters as parents
            self.q1_targets[i].load_state_dict(self.q1_nets[i].state_dict())
            self.q1_optimizers = torch.optim.Adam(self.q1_nets[i].parameters(), lr=3e-4)

            self.q2_nets[i] = QNetwork(self.observation_dimension[i], self.action_dimension[i], hidden_dim).to(self.device)
            self.q2_targets[i] = QNetwork(self.observation_dimension[i], self.action_dimension[i], hidden_dim).to(self.device)
            # Targets start with same parameters as parents
            self.q2_targets[i].load_state_dict(self.q2_nets[i].state_dict())
            self.q2_optimizers = torch.optim.Adam(self.q2_nets[i].parameters(), lr=3e-4)

    def update(self, observations: List[List[float]], actions: List[List[float]], reward: List[float], next_observations: List[List[float]], terminated: bool, truncated: bool) -> List[List[float]]:
        # For each data point type
        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
            self.replay_buffer[i].append((o, a, r, n, terminated))
            # For each element in the mini batch, take a sample from the buffer
            self.update_gradient_step(i)

    def predict(self, observations:List[List[float]], deterministic: bool = None):
        actions = []

        for i, state in enumerate(observations):
            state = torch.FloatTensor(state, device=self.device).unsqueeze(0)
            result = self.actors[i].sample(state)
            actions.append(result[0].detach().cpu().numpy()[0])
    
        self.actions = actions
        self.next_time_step()
        return actions


    def update_gradient_step(self, i:int):
        '''
        Perform the main, gradient step of the update on data point i 
        '''
        #  Sample all full batch from buffer as 2D numpy array - rows are number of batch elements, columns are state, action, rewards, next_states, terminateds
        batch = np.random.choice(self.replay_buffer[i], self.mini_batch_size)
        # Unpack into columns
        states, actions, rewards, next_states, terminateds = zip(*batch)
        
        # Convert all to Tensors of Floats
        states = torch.FloatTensor(states, device=self.device)
        actions = torch.FloatTensor(actions, device=self.device)
        # Rewards, terminated booleans are 1D
        rewards = torch.FloatTensor(rewards, device=self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states, device=self.device)
        terminateds = torch.FloatTensor(terminateds, device=self.device).unsqueeze(1)

        # Sample an action from the policy based on the next state
        new_next_actions, new_log_probs = self.actors[i].sample(next_states)

        # Update the q values from the target networks - soft update
        q1_next_target = self.q1_targets[i](next_states, new_next_actions)
        q2_next_target = self.q2_targets[i](next_states, new_next_actions)
        # Select the min of each of the target network outputs as the overall q
        q_next_target = torch.min(q1_next_target, q2_next_target)
        q_next_target = rewards + (1 - terminateds)*self.gamma*(q_next_target - self.alpha*new_log_probs)

        # Update the q networks themselves - the critics

        # Compute the Q values using the existing networks
        q1 = self.q1_nets[i](states, actions) 
        q2 = self.q2_nets[i](states, actions)
        # Compute loss for each network using MSE between q value from this network and q value from target networks
        q1_loss = F.mse_loss(q1, q_next_target.detach())
        q2_loss = F.mse_loss(q2, q_next_target.detach())
        # Reset Gradients to 0
        self.q1_optimizers[i].zero_grad()
        self.q2_optimizers[i].zero_grad()
        # Backpropogation to compute gradient
        q1_loss.backward()
        q2_loss.backward()
        # Update network parameters based on calculated gradient
        self.q1_optimizers[i].step()
        self.q2_optimizers[i].step()

        # Update the actor/policy
        new_actions, log_probs = self.actors[i].sample(states)
        q1_new_actions = self.q1_nets[i](states, new_actions) 
        q2_new_actions = self.q2_nets[i](states, new_actions)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        actor_loss = (self.alpha*log_probs - q_new_actions).mean()
        self.actor_optimizers[i].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[i].step()

        #Update target networks
        self.soft_update(self.q1_targets[i], self.q1_nets[i], self.tau)
        self.soft_update(self.q2_targets[i], self.q2_nets[i], self.tau)

    def soft_update(self, target, source, tau):
        # Iterate over target and source network parameters, zipped together
        for target_param, param in zip(target.parameters(), source.parameters()):
            # Take a small, tau sized step in direction of main networks parameter (param.data) by linearly combining the two parameters
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class PolicyNetwork(nn.Module):
    
    '''
    Neural Network representing a computationally tractable policy using a gaussian distribution. The network takes in a state, and outputs a mean and standard deviation of the given action, a continuous value, to be sampled from. Uses two hidden layers of the same size. The "actor" network.
    '''
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int=200):
        super(PolicyNetwork, self).__init__()
        # Initialize each layer individually
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state:torch.Tensor):
        '''
        Feeds forward the state through the network to return a mean and standard deviation.
        '''
        # Propogate state through the network one step at a time
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = log_std.exp()
        return mean, std

    def sample(self, state:torch.Tensor):
        '''
        Samples an action from the calculated mean and standard deviation from a Gaussian distribution, and returns the sampled action and the log probability of that action being sampled.
        '''
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        # rsample instead of sample to use the reparameterization trick - instead of sampling directly, sample from standard normal and scale and shift it
        action = dist.rsample()  
        # Get log probability for each dimension, then sum
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

class QNetwork(nn.Module):
    '''
    Nerual Network for approximating the action-value function. Estimates the expected return for taking an action in a given state and following the policy after. The "critic" network.
    '''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # Concatenate states and actions first
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value