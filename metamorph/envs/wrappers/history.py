import gym
import numpy as np

from metamorph.config import cfg
from metamorph.utils import spaces as spu


class HistoryContextWrapper(gym.Wrapper):
    """Attach a fixed-size history context slot to the observation space.

    The agent fills this slot online with a flattened history of past
    proprioceptive observations and actions.
    """

    def __init__(self, env):
        super().__init__(env)
        proprio_dim = self.observation_space["proprioceptive"].shape[0]
        action_dim = self.action_space.shape[0]
        self.history_step_dim = proprio_dim + action_dim
        self.history_dim = cfg.MODEL.HISTORY_LEN * self.history_step_dim
        self.observation_space = spu.update_obs_space(
            env, {"history_context": (self.history_dim,)}
        )

    def _zero_history(self):
        return np.zeros(self.history_dim, dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs["history_context"] = self._zero_history()
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs["history_context"] = self._zero_history()
        return obs, rew, done, info
