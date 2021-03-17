import numpy as np
import gym
from gym import spaces
from ..common import TestEnd, numpyify

class Env(gym.Env):

    def __init__(self):
        # These'd be zero-dim ideally, but that tends to upset a lot of implementations
        self.observation_space = spaces.Box(-1, +1, shape=(1,))
        self.action_space = spaces.Box(-1, +1, shape=(1,))

        self._step = 0.

    def _obs(self):
        return np.array([self._step])

    def reset(self):
        self._step = 0.
        return self._obs()

    def step(self, *args, **kwargs):
        done = self._step == 1
        reward = 3 if self._step == 0 else 2
        self._step += 1
        return self._obs(), reward, done, {}

class Test:

    def __init__(self, steps=1000, **kwargs):
        self.env = Env()
        self.gamma = .5

        self.steps = steps
        self._step = 0

    def env_fn(self, *args, **kwargs):
        return Env()

    def log(self, obs, q, **kwargs):
        self._step += 1
        from IPython import display
        display.clear_output(wait=True)
        step = numpyify(obs[:, 0])

        expected_q = np.zeros_like(step)
        expected_q[step == 0.] = 4.
        expected_q[step == 1.] = 2.

        q = numpyify(q)
        rv = ((q - expected_q)**2).mean()/(expected_q**2).mean()

        print(
            f'step\t{self._step}/{self.steps}\n'
            f'q\t{np.mean(q):.2f}±{np.std(q):.2f}\n'
            f'rv\t{rv:.0%}')
        if self._step == self.steps:
            if rv < .01:
                print('Test succeeded')
            else:
                print('Test failed')
            raise TestEnd()



