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

class ProxyAgent:

    def __call__(self, world, value=False):
        return arrdict.arrdict(
            logits=world.logits,
            v=world.v)

class RandomAgent:

    def __call__(self, world, value=True):
        B, _ = world.valid.shape
        return arrdict.arrdict(
            logits=torch.log(world.valid.float()/world.valid.sum(-1, keepdims=True)),
            actions=torch.distributions.Categorical(probs=world.valid.float()).sample(),
            v=torch.zeros((B, world.n_seats), device=world.device))

class RandomRolloutAgent:

    def __init__(self, n_rollouts):
        self.n_rollouts = n_rollouts

    def rollout(self, world):
        B, _ = world.valid.shape

        live = torch.ones((B,), dtype=torch.bool, device=world.device)
        reward = torch.zeros((B, world.n_seats), dtype=torch.float, device=world.device)
        while True:
            if not live.any():
                break

            actions = torch.distributions.Categorical(probs=world.valid.float()).sample()

            world, responses = world.step(actions)

            reward += responses.rewards * live.unsqueeze(-1).float()
            live = live & ~responses.terminal

        return reward

    def __call__(self, world, value=True):
        v = torch.stack([self.rollout(world) for _ in range(self.n_rollouts)]).mean(0)
        return arrdict.arrdict(
            logits=torch.log(world.valid.float()/world.valid.sum(-1, keepdims=True)),
            actions=torch.distributions.Categorical(probs=world.valid.float()).sample(),
            v=v)

def uniform_logits(valid):
    return torch.log(valid.float()/valid.sum(-1, keepdims=True))
    
class InstantWin(arrdict.namedarrtuple(fields=('envs',))):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.envs, torch.Tensor):
            # Need this conditional to deal with the case where we're calling a method like `self.clone()`, and the
            # intermediate arrdict generated is full of methods, which will break this here init function.
            return 

        self.device = self.envs.device
        self.n_envs = len(self.envs)
        self.n_seats = 1

        self.obs_space = (0,)
        self.action_space = (1,)
    
        self.valid = torch.ones((self.n_envs, 1), dtype=torch.bool, device=self.device)
        self.seats = torch.zeros((self.n_envs,), dtype=torch.long, device=self.device)

        self.logits = uniform_logits(self.valid)
        self.v = torch.ones((self.n_envs, self.n_seats), dtype=torch.float, device=self.device)

    def step(self, actions):
        trans = arrdict.arrdict(
            terminal=torch.ones((self.n_envs,), dtype=torch.bool, device=self.device),
            rewards=torch.ones((self.n_envs, self.n_seats), dtype=torch.float, device=self.device))
        return self, trans

def instant_win(n_envs=1, device='cuda'):
    return InstantWin(envs=torch.arange(n_envs, device=device))

class FirstWinsSecondLoses:

    def __init__(self, n_envs=1, device='cuda'):
        self.device = torch.device(device)
        self.n_envs = n_envs
        self.n_seats = 2

        self.obs_space = (0,)
        self.action_space = (1,)

        self._seats = torch.zeros((self.n_envs,), dtype=torch.long, device=self.device)

    def _observe(self):
        valid = torch.ones((self.n_envs, 1), dtype=torch.bool, device=self.device)
        return arrdict.arrdict(
            valid=valid,
            seats=self._seats,
            logits=uniform_logits(valid),
            v=torch.tensor([[+1., -1.]], device=self.device).expand(self.n_envs, 2)).clone()

    def reset(self):
        return self._observe()

    def step(self, actions):
        terminal = (self._seats == 1)
        responses = arrdict.arrdict(
            terminal=terminal,
            rewards=torch.stack([terminal.float(), -terminal.float()], -1))

        self._seats += 1
        self._seats[responses.terminal] = 0

        inputs = self._observe()
        
        return responses, inputs

    def state_dict(self):
        return arrdict.arrdict(
            seat=self._seats).clone()

    def load_state_dict(self, sd):
        self._seats[:] = sd.seat

class AllOnes:

    def __init__(self, n_envs=1, length=4, device='cpu'):
        self.device = device 
        self.n_envs = n_envs
        self.n_seats = 1
        self.length = length

        self.obs_space = (0,)
        self.action_space = (2,)
        self.history = torch.full((n_envs, length), -1, dtype=torch.long, device=self.device)
        self.idx = torch.full((n_envs,), 0, dtype=torch.long, device=self.device)

    def _observe(self):
        correct_so_far = (self.history == 1).sum(-1) == self.idx+1
        correct_to_go = 2**((self.history == 1).sum(-1) - self.length).float()
        v = correct_so_far.float()*correct_to_go

        valid = torch.ones((self.n_envs, 2), dtype=torch.bool, device=self.device)
        return arrdict.arrdict(
            valid=valid,
            seats=torch.zeros((self.n_envs,), dtype=torch.int, device=self.device),
            logits=uniform_logits(valid),
            v=v[..., None]).clone()

    def reset(self):
        return self._observe()

    def step(self, actions):
        self.history[:, self.idx] = actions

        inputs = self._observe()

        self.idx += 1 

        response = arrdict.arrdict(
            terminal=(self.idx == self.length),
            rewards=((self.idx == self.length) & (self.history == 1).all(-1))[:, None].float())

        self.idx[response.terminal] = 0
        self.history[response.terminal] = -1
        
        return response, inputs

    def state_dict(self):
        return arrdict.arrdict(
            history=self.history,
            idx=self.idx).clone()

    def load_state_dict(self, d):
        self.history = d.history
        self.idx = d.idx

