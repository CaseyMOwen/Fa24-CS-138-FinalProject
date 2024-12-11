"""
Q Learning Implementation
"""

from typing import Any, List, Tuple
import numpy as np
from .baseagent import Agent
from citylearn.citylearn import CityLearnEnv

class QLearning(Agent):
    """ Assumed passed envirionment is discretized """

    
    def __init__(
        self, 
        env: CityLearnEnv, 
        epsilon: float = 0.1, 
        epsilon_decay: bool = True,
        min_epsilon: float = 0.01,
        learning_rate: float = 0.01, 
        discount_factor: float = 0.9, 
        **kwargs: Any
    ):
    
        super().__init__(env, **kwargs)
        self.starting_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table, self.explore_counts, self.exploit_counts = self.__initialize_q()
        self.predictions = []

    def __greedy_action(
        self, 
        observations: List[List[float]]
    ) -> List[List[float]]:

        """
        Return an action with the greatest Q-value for the given observation.
        
        Observations is a list of 1 sublist.
        The sublist contains discretized observation of all buildings
        """

        actions = []
        
        for index, observation in enumerate(observations):
            observation = observation[0]

            best_action = np.argmax(self.q_table[index][observation])
            if (self.q_table[index][observation, best_action] == 0):
                best_action = self.action_space[index].sample()

            actions.append([best_action])
        
        return actions


    def __initialize_q(self) -> np.ndarray:
        """
        Initialize values for all (observation, action) pairs in Q table to 0.
        
        Q table has shape (1, (O.n, A.n)) where
        O.n is the number of possible observation value and A.n is the number
        of possible action values.
        """

        q = []
        exploration = []
        exploitation = []
        obs_action_pairs = zip(self.observation_space, self.action_space)

        for obs, action in obs_action_pairs:
            shape = (obs.n, action.n)
            q.append(np.zeros(shape=shape))
            exploration.append(np.zeros(shape=shape))
            exploitation.append(np.zeros(shape=shape))
        
        return q, exploration, exploitation

    def update(
        self,
        observations: List[List[float]],
        actions: List[List[float]],
        reward: List[float],
        next_observations: List[List[float]],
        terminated: bool,
        truncated: bool,
    ):
        """
        observations : 
            List of sublists. Only one sublist in list and it contains all observations for all buildings

        actions :
            List of sublists. When central agent = True, there is one sublist
            and it contains action for all buildings.
        reward :
            List of reward for current time step.

        next_observations :
            List of sublists for observations of next time step. Follows structure
            of observations parameter.

        terminated : bool
            Indicates that episode has ended.
        
        truncated : bool
            Indicates if episode was cut short

        Note:
        terminated and truncated are ignored but are kept as parameters to
        maintain compatibility with environment
        """

        packaged_timestep = zip(observations, actions, reward, next_observations)
        for index, (o, a, r, next_o) in enumerate(packaged_timestep):
            o = o[0]
            a = a[0]
            next_o = next_o[0]

            q_t = self.q_table[index][o, a]
            q_t2_max = np.argmax(self.q_table[index][next_o])

            q_update = self.learning_rate * (r + self.discount_factor*q_t2_max - q_t)
            self.q_table[index][o, a] = q_t + q_update
            
            if self.__explored:
                self.explore_counts[index][o, a] += 1
            else:
                self.exploit_counts[index][o, a] += 1


    def predict(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """Return action for given observation.
        
        When deterministic is True, action returned wil return action with 
        greatest Q_value in.

        When deterministic is False, action returned will be randomly sampled.
        """

        is_greedy = np.random.random() > self.epsilon
        
        if deterministic or is_greedy:
            actions = self.__greedy_action(observations)
            self.__explored = False
        else:
            actions = [[s.sample()] for s in self.action_space]
            self.__explored = True

        if self.epsilon_decay:
            curr_episode = int(self.time_step / self.episode_time_steps)
            self.epsilon = max(self.starting_epsilon ** (curr_episode + 1), self.min_epsilon)
            
        self.actions = actions
        self.next_time_step()
        self.predictions.append(actions)

        return actions




