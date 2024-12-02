'''
Custom Sarsa Implementation. Taking much of structural organization of an agent from CityLearn Q Learning
'''


import math
from typing import Any, List, Tuple
import numpy as np
from .baseagent import Agent
from citylearn.citylearn import CityLearnEnv
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib


class Sarsa(Agent):
    def __init__(
        self, env: CityLearnEnv, epsilon: float = 0.1, 
        alpha: float = 0.5, gamma: float = 0.95, **kwargs: Any,
    ):
        super().__init__(env, **kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = self.init_q_table()
        self.previous_actions = None
        self.previous_observations = None

    def predict(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """Provide actions for current time step.

        If `deterministic` = True or, randomly generated number is greater than `epsilon`, return
        deterministic action from Q-Table i.e. action with max Q-value for given observations 
        otherwise, return randomly sampled action.
        
        Parameters
        ----------
        observations: List[List[float]]
            Environment observations
        deterministic: bool, default: False
            Wether to return purely exploitatative deterministic actions.

        Returns
        -------
        actions: List[List[float]]
            Action values
        """
        nprs = np.random.RandomState(self.random_seed + self.time_step)
        if deterministic or nprs.random() > self.epsilon:
            # Use q-function to decide action
            actions = self.choose_best_actions(observations)
        else:
            # Explore random action
            actions = [[s.sample()] for s in self.action_space]

        # self.actions = actions
        self.next_time_step()
        return actions
    
    def choose_best_actions(self, observations):
        actions = []

        for i, o in enumerate(observations):
            if not self.env.central_agent:
                raise ValueError("Sarsa can only be implemented on a central agent")
            else:
                '''
                From documentation on central agent:

                "If central_agent is True, a list of 1 sublist containing all building observation names is returned in the same order as buildings. The shared_observations names are only included in the first building’s observation names. If central_agent is False, a list of sublists is returned where each sublist is a list of 1 building’s observation names and the sublist in the same order as buildings"

                We are enforcing central_agent to be true, so this is a list of only one sublist.
                '''
                # Discrete wrapper condenses continuous observations of many values into a single integer representing a unique combo of available bins, so each list is only one element long
                obs_idx = o[0]
                # Slicing q table so this is array of all action values at the given observation
                act_vals = self.q_table[i][obs_idx]
                # If all values for this observation are nan (unvisited), choose randomly
                if np.isnan(act_vals).all():
                    action = self.action_space[i].sample()
                else:
                    action = np.nanargmax(act_vals)
                actions.append(action)
        return actions

    def update(self, observations: List[List[float]], actions: List[List[float]], reward: List[float], next_observations: List[List[float]], terminated: bool, truncated: bool):
        r"""Update Q-Table using Bellman equation.

        Parameters
        ----------
        observations : List[List[float]]
            Previous time step observations.
        actions : List[List[float]]
            Previous time step actions.
        reward : List[float]
            Current time step reward.
        next_observations : List[List[float]]
            Current time step observations.
        terminated : bool
            Indication that episode has ended.
        truncated : bool
            If episode truncates due to a time limit or a reason that is not defined as part of the task MDP.
        """
        if not self.previous_actions or not self.previous_observations:
            self.previous_actions = actions
            self.previous_observations = observations
            return

        
        # Compute temporal difference target and error to udpate q-function

        for i, (o, a, r) in enumerate(zip(observations, actions, reward)):
            # Current obs and current action
            cur_o, cur_a = o[0], a[0]
            last_o = self.previous_observations[0]
            last_a = self.previous_actions[0]
            last_q = self.q_table[i][last_o, last_a]
            last_q = 0.0 if math.isnan(last_q) else last_q
            current_q = self.q_table[i][cur_o, cur_a]
            current_q = 0.0 if math.isnan(current_q) else current_q
            
            # update q
            new_last_q = last_q + self.alpha*(r + self.gamma*current_q - last_q)
            self.q_table[i][last_o, last_a] = new_last_q
        
    def init_q_table(self):
        '''
        Ensure that the q table has been initialized within the given state. Alternative to building q table with all possible states at the beginning - build it on the fly instead, since many possible states have very low probability of being reached
        '''
        q = [None]*len(self.observation_space)
        # i is each controllable element - which has one observation and one action each step
        for i, (od, ad) in enumerate(zip(self.observation_space, self.action_space)):
            # .n is the number of discrete elements of the Box
            # So each elements q table has rows for each discrete observation, and columns for each discrete actions. Both are discretized using gymnasium.spaces.Box if continuous.
            shape = (od.n, ad.n)
            q[i] = np.full(shape=shape, fill_value=np.nan)

        return q
