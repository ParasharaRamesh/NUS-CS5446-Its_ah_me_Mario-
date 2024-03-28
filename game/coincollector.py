'''
Idea is to build a custom reward function to include the score as well.

Need to somehow get the score gained between time steps as a reward.
'''
from gym_super_mario_bros import SuperMarioBrosEnv
import gym
from utils.constants import COIN_COLLECTOR_ENV_REWARD_RANGE


class CoinCollectorSuperMarioBrosEnv(SuperMarioBrosEnv):
    #score btn 2 time frames can maybe go upto 8000 so we can just divide by 100 (reference https://www.mariowiki.com/Point)
    reward_range = COIN_COLLECTOR_ENV_REWARD_RANGE

    def __init__(self):
        super().__init__()

        # variable to keep track of score deltas
        self._score_last = 0

    @property
    def _score_reward(self):
        _reward = self._score - self._score_last
        self._score_last = self._score
        return _reward/100

    # This should override the parent function
    def _get_reward(self):
        return self._x_reward + self._score_reward + self._time_penalty + self._death_penalty

