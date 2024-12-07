"""
Q Learning Implementation
"""

from typing import Any, List, Tuple
import numpy as np
from .baseagent import Agent
from citylearn.citylearn import CityLearnEnv

class QLearning(Agent):
    def __init__(
        self, 
        env: CityLearnEnv, 
        epsilon: float = 0.1, 
        learning_rate: float = 0.01, 
        discount_factor: float = 0.9, 
        **kwargs: Any
    ):
    
        super().__init__(env, **kwargs)
        self.epsilon = epsilon
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
        
        When central_agent is True, observations is a list of 1 sublist.
        The sublist contains observation of all buildings

        When central agent is False, observations is a list of n sublists
        where n is number of buildings.
        Each sublist contains observation of corresponding building.
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
        
        When central_agent is True, Q table has shape (1, (O.n, A.n)) where
        O.n is the number of possible observation value and A.n is the number
        of possible action values.

        When central_agent is False, Q table has shape(Element, (O_e.n, A_e.n))
        where Element is the observation type (e.g indoor bulb temperature),
        O_e.n is the number of possible observation values for the Element and
        A_e.n is the number of possible action values for the Element.
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
            List of sublists. When central agent = True, there is only one
            sublist and it contains all observations in order of buildings.
            When central agent is false, there is a sublist for each building
            containing observations for corresponding building.
        
        actions :
            List of sublists. When central agent = True, there is one sublist
            and it contains all actions in order of buildings.
            When central agent is false, there is a sublisst for each building
            containing actions for corresponding building.

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

        self.actions = actions
        self.next_time_step()
        self.predictions.append(actions)

        return actions




