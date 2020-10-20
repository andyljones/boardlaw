"""
What to test?
    * Does it all run? Only need one action, one player, one timestep for this. 
    * Does it correctly pick the winning move? One player, two actions, instant returns. 
    * Does it correctly pick the winning move deep in the tree? One player, two actions, a bunch of ignored actions,
      and then a payoff depending on those first two
    * Does it correctly estimate state values for different players? Need two players with different returns.
    * 
"""
import torch
from rebar import arrdict
from . import heads
from collections import namedtuple

class RandomAgent:

    def __call__(self, inputs, value=False):
        decisions = arrdict.arrdict(
            logits=torch.log(inputs.valid.float()/inputs.valid.sum(-1, keepdims=True)))
        if value:
            decisions['v'] = torch.zeros_like(decisions.logits[:, 0])
        return decisions

class KnownValueRandomAgent:

    def __call__(self, inputs, value=True):
        return arrdict.arrdict(
            logits=torch.log(inputs.valid.float()/inputs.valid.sum(-1, keepdims=True)),
            v=inputs.v)

class RandomRolloutAgent:

    def __init__(self, env, n_rollouts):
        self.env = env
        self.n_rollouts = n_rollouts

    def rollout(self, inputs):
        env = self.env
        original = env.state_dict()
        chooser = inputs.seat

        live = torch.ones_like(inputs.terminal)
        reward = torch.zeros_like(inputs.terminal, dtype=torch.float)
        while True:
            if not live.any():
                break

            actions = torch.distributions.Categorical(probs=inputs.mask.float()).sample()
            same_seat = (inputs.seat == chooser).float()

            inputs = env.step(actions)

            reward += inputs.reward * live.float() * same_seat
            live = live & ~inputs.terminal

        env.load_state_dict(original)

        return reward

    def __call__(self, inputs, value=True):
        B, A = inputs.valid.shape
        return arrdict.arrdict(
            logits=torch.log(inputs.valid.float()/inputs.valid.sum(-1, keepdims=True)),
            v=torch.stack([self.rollout(inputs) for _ in range(self.n_rollouts)]).mean(0))
    
class InstantWin:

    def __init__(self, n_envs=1, device='cuda'):
        self.device = torch.device(device)
        self.n_envs = n_envs
        self.n_seats = 1

        self.obs_space = (0,)
        self.action_space = (1,)

    def reset(self):
        return arrdict.arrdict(
            valid=torch.ones((self.n_envs, 1), dtype=torch.bool, device=self.device),
            seats=torch.zeros((self.n_envs,), dtype=torch.long, device=self.device))

    def step(self, actions):
        responses = arrdict.arrdict(
            terminal=torch.ones((self.n_envs,), dtype=torch.bool, device=self.device),
            rewards=torch.zeros((self.n_envs, self.n_seats), dtype=torch.float, device=self.device))
        
        inputs = arrdict.arrdict(
            valid=torch.ones((self.n_envs, 1), dtype=torch.bool, device=self.device),
            seats=torch.zeros((self.n_envs,), dtype=torch.long, device=self.device))
        
        return responses, inputs

    def state_dict(self):
        return arrdict.arrdict()

    def load_state_dict(self, sd):
        pass

class FirstWinsSecondLoses:

    def __init__(self, n_envs=1, device='cuda'):
        self.device = torch.device(device)
        self.n_envs = n_envs
        self.n_seats = 2

        self.obs_space = (0,)
        self.action_space = (1,)

        self._seats = torch.zeros((self.n_envs,), dtype=torch.long, device=self.device)

    def reset(self):
        return arrdict.arrdict(
            valid=torch.ones((self.n_envs, 1), dtype=torch.bool, device=self.device),
            seats=self._seats).clone()

    def step(self, actions):
        terminal = (self._seats == 1)
        responses = arrdict.arrdict(
            terminal=terminal,
            rewards=torch.stack([terminal.float(), -terminal.float()], -1))
        self._seats[responses.terminal] = 0
        
        inputs = arrdict.arrdict(
            valid=torch.ones((self.n_envs, 1), dtype=torch.bool, device=self.device),
            seats=self._seats).clone()
        
        return responses, inputs

    def state_dict(self):
        return arrdict.arrdict(
            seat=self._seats).clone()

    def load_state_dict(self, sd):
        self._seats[:] = sd.seat

class BinaryTree:

    def __init__(self, n_envs=1, length=2, device='cpu'):
        self.device = device 
        self.n_envs = n_envs
        self.length = length

        self.action_space = namedtuple('Vector', ('dim',))((2,))
        self.history = torch.full((n_envs, length), -1, dtype=torch.long, device=self.device)
        self.idx = torch.full((n_envs,), 0, dtype=torch.long, device=self.device)

    def _observe(self):
        return arrdict.arrdict(
            seat=torch.zeros((self.n_envs,), dtype=torch.int, device=self.device),
            mask=torch.ones((self.n_envs, 2), dtype=torch.bool, device=self.device),
            terminal=(self.idx == self.length),
            reward=(self.idx == self.length) & (self.history == 1).all(-1))

    def reset(self):
        return self._observe()

    def step(self, actions):
        self.history[:, self.idx] = actions

        self.idx += 1 

        inputs = self._observe().clone()

        self.idx[inputs.terminal] = 0
        self.history[inputs.terminal] = -1
        
        return inputs

    def state_dict(self):
        return arrdict.arrdict(
            history=self.history,
            idx=self.idx).clone()

    def load_state_dict(self, d):
        self.history = d.history
        self.idx = d.idx
