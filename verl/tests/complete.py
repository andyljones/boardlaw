from torch import nn
import torch
import numpy as np
import gym
from gym import spaces
from ..common import TestEnd, numpyify

def final_reward(action, target):
    return (1 - 2*abs(action - target))

class Env(gym.Env):

    def __init__(self):
        # These'd be zero-dim ideally, but that tends to upset a lot of implementations
        self.observation_space = spaces.Box(-1, +1, shape=(3,))
        self.action_space = spaces.Box(-1, +1, shape=(1,))

        self._target = np.nan
        self._action = np.nan
        self._step = 0

    def _obs(self):
        return np.array([self._target, self._action, self._step])

    def reset(self):
        self._step = 0
        self._action = 0.
        self._target = np.random.choice([-1/2., +1/2.])
        return self._obs()

    def step(self, action):
        [action] = action
        if self._step == 0:
            self._action = action
        done = self._step == 1
        reward = final_reward(self._action, self._target)*float(done)
        self._step += 1
        return self._obs(), reward, done, {}

class PlantedQ(nn.Module):

    def forward(self, obs):
        target, fst_action, step = obs['obs'].T
        action = obs['act'].squeeze(-1)
        action[step == 1] = fst_action
        reward = final_reward(action, target)
        reward[step == 0] = reward/2.
        return reward

class PlantedPi(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = 'cuda'

    def forward(self, obs):
        target, action, step = obs[None, :].T
        return target

class PlantedNetwork(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.q = PlantedQ()
        self.pi = PlantedPi()

    def act(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(self.pi.device).float()
            return self.pi(obs).cpu().numpy()

class Test:

    def __init__(self, steps=1000, **kwargs):
        self.env = Env()
        self.network = PlantedNetwork()
        self.gamma = .5

        self.steps = steps
        self._step = 0

    def env_fn(self, *args, **kwargs):
        return Env()

    def log(self, obs, action, q=np.nan, **kwargs):
        self._step += 1

        target, obs_action, step = numpyify(obs.T)
        q = numpyify(q)

        action = numpyify(action[:, 0])
        action[step == 1] = obs_action[step == 1]

        reward = final_reward(action, target)

        expected_q = np.zeros_like(reward)
        expected_q[step == 0] = self.gamma*reward[step == 0]
        expected_q[step == 1] = reward[step == 1]
        q_resid_var = ((expected_q - q)**2).mean()/(expected_q**2).mean()

        a_resid_var = ((target[step == 0] - action[step == 0])**2).mean()/(target[step == 0]**2).mean()

        from IPython import display
        display.clear_output(wait=True)
        print(
            f'step\t{self._step}/{self.steps}\n'
            f'a\t{np.mean(action):+.2f}±{np.std(action):.2f}\n'
            f'v\t{np.mean(q):+.2f}±{np.std(q):.2f}\n'
            f'r\t{np.mean(reward):+.2f}\n'
            f'arv\t{a_resid_var:+.0%}\n'
            f'qrv\t{q_resid_var:+.0%}')
        if self._step == self.steps:
            if a_resid_var < .01:
                print('Test succeeded')
            else:
                print('Test failed')
            raise TestEnd()





