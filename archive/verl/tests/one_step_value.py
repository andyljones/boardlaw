import numpy as np
import gym
from gym import spaces
from ..common import TestEnd, numpyify

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
        return self._obs(), 1., True, {}

class Test:

    def __init__(self, steps=1000, **kwargs):
        self.env = Env()

        self.steps = steps
        self._step = 0

    def env_fn(self, *args, **kwargs):
        return Env()

    def log(self, q, **kwargs):
        self._step += 1
        from IPython import display
        display.clear_output(wait=True)
        q = numpyify(q)

        print(
            f'step\t{self._step}/{self.steps}\n'
            f'q\t{np.mean(q):.2f}±{np.std(q):.2f}')
        if self._step == self.steps:
            if abs(q - 1) < .1:
                print('Test succeeded')
            else:
                print('Test failed')
            raise TestEnd()


