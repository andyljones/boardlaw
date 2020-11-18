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

    @classmethod
    def initial(cls, n_envs=1, device='cuda'):
        return cls(envs=torch.arange(n_envs, device=device))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.envs, torch.Tensor):
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

class FirstWinsSecondLoses(arrdict.namedarrtuple(fields=('seats',))):

    @classmethod
    def initial(cls, n_envs=1, device='cuda'):
        return cls(seats=torch.zeros(n_envs, device=device, dtype=torch.int))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.seats, torch.Tensor):
            return 

        self.device = self.seats.device
        self.n_envs = len(self.seats)
        self.n_seats = 2

        self.obs_space = (0,)
        self.action_space = (1,)
    
        self.valid = torch.ones((self.n_envs, 1), dtype=torch.bool, device=self.device)
        self.seats = self.seats

        self.logits = uniform_logits(self.valid)
        self.v = torch.stack([torch.ones_like(self.seats), -torch.ones_like(self.seats)], -1).float()

    def step(self, actions):
        terminal = (self.seats == 1)
        trans = arrdict.arrdict(
            terminal=terminal,
            rewards=torch.stack([terminal.float(), -terminal.float()], -1))
        return type(self)(seats=1-self.seats), trans


class AllOnes(arrdict.namedarrtuple(fields=('history', 'count'))):

    @classmethod
    def initial(cls, n_envs=1, n_seats=1, length=4, device='cuda'):
        return cls(
            history=torch.full((n_envs, length, n_seats), -1, dtype=torch.long, device=device),
            count=torch.full((n_envs,), 0, dtype=torch.long, device=device))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.count, torch.Tensor):
            return 

        self.n_envs, self.length, self.n_seats = self.history.shape[-3:]
        self.device = self.count.device 

        self.max_count = self.n_seats * self.length

        self.obs_space = heads.Tensor((1,))
        self.action_space = heads.Masked(2)

        self.valid = torch.ones(self.count.shape + (2,), dtype=torch.bool, device=self.device)
        self.seats = self.count % self.n_seats

        self.obs = self.count[..., None].float()/self.max_count

        self.envs = torch.arange(self.n_envs, device=self.device)

        # Planted values for validation use
        self.logits = uniform_logits(self.valid)

        correct_so_far = (self.history == 1).sum(-1) == self.count[..., None]
        correct_to_go = 2**((self.history == 1).sum(-1) - self.length).float()

        v = correct_so_far.float()*correct_to_go
        self.v = v[..., None]

    def step(self, actions):
        history = self.history.clone()
        idx = self.count//self.n_seats
        history[self.envs, idx, self.seats] = actions
        count = self.count + 1 

        terminal = (count == self.max_count)
        reward = ((count == self.max_count)[:, None] & (history == 1).all(-2)).float()
        transition = arrdict.arrdict(terminal=terminal, rewards=reward)

        count[terminal] = 0
        history[terminal] = -1
        
        world = type(self)(
            history=history,
            count=count)
        return world, transition

def test_all_ones():
    game = AllOnes.initial(n_envs=3, n_seats=2, length=3, device='cpu')
    for s in range(6):
        action = game.seats % 2
        game, transition = game.step(action.long())

    assert transition.terminal.all()
    assert (transition.rewards[:, 0] == 0).all()
    assert (transition.rewards[:, 1] == 1).all()
