
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from typing import Any
import numpy as np

class ReducedCostReward(RewardFunction):
    def __init__(self, env_metadata: dict[str, Any]):
        """Initialize CustomReward.

        Parameters
        ----------
        env_metadata: dict[str, Any]:
            General static information about the environment.
        """

        super().__init__(env_metadata)

    def calculate(self, observations: list[dict[str, int | float]]) -> list[float]:
        reward = 0
        for o, m in zip(observations, self.env_metadata['buildings']):
            cost = o['net_electricity_consumption']*o['electricity_pricing']
            battery_soc = o['electrical_storage_soc']
            penalty = -(1.0 + np.sign(cost)*battery_soc)
            reward = penalty*abs(cost)

            reward += (reward)
        
        return [reward]