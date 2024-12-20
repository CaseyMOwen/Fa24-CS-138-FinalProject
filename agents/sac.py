'''
Custom Implementation of SAC agent for use with CityLearn. Uses code from sources linked below.
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
        self, env: CityLearnEnv, mini_batch_size:int=5, gamma:float=.99, tau:float=.005, **kwargs: Any,
    ):
        '''
        Agent implementation of a Soft-Actor Critic Agent with automatic entropy adjustment, for use with CityLearn environment.
        '''
        super().__init__(env, **kwargs)
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.tau = tau
        self.action_scaling_coefficient = 0.5



        self.replay_buffer = [[] for _ in self.action_space]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.observation_dimension = [len(names) for names in self.observation_names]
        self.init_networks(50)

    def init_networks(self, hidden_dim:int):
        '''
        Initializes each neural network as a list of networks, one for each observation varaible. Uses a learning rate of 3e-4.
        '''
        learning_rate = .0003
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
        
        #Entropy
        self.alphas =  [None for _ in self.action_space]
        self.target_entropys =  [None for _ in self.action_space]
        self.log_alphas =  [None for _ in self.action_space]
        self.alpha_optimizers =  [None for _ in self.action_space]


        for i in range(len(self.action_space)):
            self.actors[i] = PolicyNetwork(self.observation_dimension[i], self.action_dimension[i], self.action_space[i], self.action_scaling_coefficient, hidden_dim).to(self.device)
            self.actor_optimizers[i] = torch.optim.Adam(self.actors[i].parameters(), lr=learning_rate)

            self.q1_nets[i] = QNetwork(self.observation_dimension[i], self.action_dimension[i], hidden_dim).to(self.device)
            self.q1_targets[i] = QNetwork(self.observation_dimension[i], self.action_dimension[i], hidden_dim).to(self.device)
            # Targets start with same parameters as parents
            self.q1_targets[i].load_state_dict(self.q1_nets[i].state_dict())
            self.q1_optimizers[i] = torch.optim.Adam(self.q1_nets[i].parameters(), lr=learning_rate)

            self.q2_nets[i] = QNetwork(self.observation_dimension[i], self.action_dimension[i], hidden_dim).to(self.device)
            self.q2_targets[i] = QNetwork(self.observation_dimension[i], self.action_dimension[i], hidden_dim).to(self.device)
            # Targets start with same parameters as parents
            self.q2_targets[i].load_state_dict(self.q2_nets[i].state_dict())
            self.q2_optimizers[i] = torch.optim.Adam(self.q2_nets[i].parameters(), lr=learning_rate)

            self.alphas[i] = 1
            self.target_entropys[i] = -np.prod(len(self.action_space)).item()  # heuristic value
            self.log_alphas[i] = torch.zeros(1, requires_grad=True).to(self.device)
            self.alpha_optimizers[i] = torch.optim.Adam([self.log_alphas[i]], lr=learning_rate)

    def update(self, observations: List[List[float]], actions: List[List[float]], reward: List[float], next_observations: List[List[float]], terminated: bool, truncated: bool) -> List[List[float]]:
        '''
        Updates the agent according to obsercations, actions, a reward, and the observations at the next state, as well as information about whether the episode has ended or is truncated.
        '''
        # For each data point type
        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
            self.replay_buffer[i].append((o, a, r, n, terminated))
            # For each element in the mini batch, take a sample from the buffer
            self.update_gradient_step(i)

    def predict(self, observations:List[List[float]], deterministic: bool = None):
        '''
        Determines a set of actions based on a set of observations.
        '''
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
        Perform the main, gradient step of the update on data point i. Core SAC algorithm
        '''
        #  Sample all full batch from buffer as 2D numpy array - rows are number of batch elements, columns are state, action, rewards, next_states, terminateds
        batch = random.choices(self.replay_buffer[i], k=self.mini_batch_size)
        # Unpack into columns
        states, actions, rewards, next_states, terminateds = zip(*batch)
        
        # Convert all to Tensors of Floats
        states = torch.FloatTensor(states, device=self.device)
        actions = torch.FloatTensor(actions, device=self.device)
        # Rewards, terminated booleans are 1D
        rewards = torch.FloatTensor(rewards, device=self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states, device=self.device)
        terminateds = torch.FloatTensor(terminateds, device=self.device).unsqueeze(1)

        with torch.no_grad():
            # Sample an action from the policy based on the next state
            new_next_actions, new_log_probs = self.actors[i].sample(next_states)

            # Update the q values from the target networks - soft update
            q1_next_target = self.q1_targets[i](next_states, new_next_actions)
            q2_next_target = self.q2_targets[i](next_states, new_next_actions)
            # Select the min of each of the target network outputs as the overall q
            q_next_target = torch.min(q1_next_target, q2_next_target)
            q_next_target = rewards + (1 - terminateds)*self.gamma*(q_next_target - self.alphas[i]*new_log_probs)

        # Update the q networks themselves - the critics

        # Compute the Q values using the existing networks
        q1 = self.q1_nets[i](states, actions) 
        q2 = self.q2_nets[i](states, actions)
        # Compute loss for each network using MSE between q value from this network and q value from target networks
        q1_loss = nn.SmoothL1Loss()(q1, q_next_target.detach())
        q2_loss = nn.SmoothL1Loss()(q2, q_next_target.detach())
        # Reset Gradients to 0
        self.q1_optimizers[i].zero_grad()
        self.q2_optimizers[i].zero_grad()
        # Backpropogation to compute gradient
        q1_loss.backward()
        q2_loss.backward()
        # Clip the gradient to help with convergence
        torch.nn.utils.clip_grad_norm_(self.q1_nets[i].parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.q2_nets[i].parameters(), 1)
        # Update network parameters based on calculated gradient
        self.q1_optimizers[i].step()
        self.q2_optimizers[i].step()

        # Update the actor/policy
        new_actions, log_probs = self.actors[i].sample(states)
        q1_new_actions = self.q1_nets[i](states, new_actions) 
        q2_new_actions = self.q2_nets[i](states, new_actions)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        actor_loss = (self.alphas[i]*log_probs - q_new_actions).mean()
        self.actor_optimizers[i].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1)
        self.actor_optimizers[i].step()

        #Update entropy parameter 
        alpha_loss = (self.log_alphas[i] * (-log_probs - self.target_entropys[i]).detach()).mean()
        self.alpha_optimizers[i].zero_grad()
        alpha_loss.backward()
        self.alpha_optimizers[i].step()
        self.alphas[i] = self.log_alphas[i].exp()


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
    def __init__(self, state_dim:int, action_dim:int, action_space, action_scaling_coef, hidden_dim:int=200):
        super(PolicyNetwork, self).__init__()
        # Initialize each layer individually
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.action_scale = torch.FloatTensor(
        action_scaling_coef * (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
        action_scaling_coef * (action_space.high + action_space.low) / 2.)
        self.epsilon = 1e-6

    def forward(self, state:torch.Tensor):
        '''
        Feeds forward the state through the network to return a mean and standard deviation.
        '''
        # Propogate state through the network one step at a time
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x1))
        mean = self.mean(x2)
        log_std = self.log_std(x2)
        log_std = torch.clamp(log_std, min=-20, max=2) #Avoid -inf when std -> 0
        std = log_std.exp()
        return mean, std

    def sample(self, state:torch.Tensor):
        '''
        Samples an action from the calculated mean and standard deviation from a Gaussian distribution, and returns the sampled action and the log probability of that action being sampled.
        '''
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        # rsample instead of sample to use the reparameterization trick - instead of sampling directly, sample from standard normal and scale and shift it
        # action = dist.rsample()  
        # # Get log probability for each dimension, then sum
        # log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

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