import numpy as np
import gym
from gym import spaces
from ..common import TestEnd

class Env(gym.Env):

    def __init__(self):
        # These'd be zero-dim ideally, but that tends to upset a lot of implementations
        self.observation_space = spaces.Box(-1, +1, shape=(1,))
        self.action_space = spaces.Box(-1, +1, shape=(1,))

    def _obs(self):
        return np.zeros((1,))

    def reset(self):
        return self._obs()

    def step(self, *args, **kwargs):
        return self._obs(), 0., True, {}

class Test:

    def __init__(self, steps=3, **kwargs):
        self.env = Env()

        self.steps = steps
        self._step = 0

    def env_fn(self, *args, **kwargs):
        return Env()

    def log(self, count, **kwargs):
        self._step += count
        from IPython import display
        display.clear_output(wait=True)
        print(f'Step: {self._step}/{self.steps}')
        if self._step == self.steps:
            print('Crash test finished successfully')
            raise TestEnd()

