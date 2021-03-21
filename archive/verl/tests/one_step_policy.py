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

    def step(self, action):
        [action] = action
        reward = 1 - 2*abs(action - 1/2)
        return self._obs(), reward, True, {}

class Test:

    def __init__(self, steps=1000, **kwargs):
        self.env = Env()

        self.steps = steps
        self._step = 0

    def env_fn(self, *args, **kwargs):
        return Env()

    def log(self, action, q=np.nan, **kwargs):
        self._step += 1
        action = numpyify(action.squeeze(-1))
        q = numpyify(q)

        expected_q = 1 - 2*abs(action - 1/2)
        resid_var = ((expected_q - q)**2).mean()/(expected_q**2).mean()

        from IPython import display
        display.clear_output(wait=True)
        print(
            f'step\t{self._step}/{self.steps}\n'
            f'a\t{np.mean(action):+.2f}±{np.std(action):.2f}\n'
            f'q\t{np.mean(q):+.2f}±{np.std(q):.2f}\n'
            f'rv\t{resid_var:.0%}')
        if self._step == self.steps:
            if abs(action - 1/2).max() < .1:
                print('Test succeeded')
            else:
                print('Test failed')
            raise TestEnd()



