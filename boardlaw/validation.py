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
from pavlov import stats
from . import heads
import numpy as np

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

class MonteCarloAgent:

    def __init__(self, n_rollouts, temperature=1.):
        self.n_rollouts = n_rollouts
        self.temperature = temperature

    def rollout(self, world):
        B, _ = world.valid.shape

        live = torch.ones((B,), dtype=torch.bool, device=world.device)
        reward = torch.zeros((B, world.n_seats), dtype=torch.float, device=world.device)
        first_actions = None
        while True:
            if not live.any():
                break

            actions = torch.distributions.Categorical(probs=world.valid.float()).sample()
            if first_actions is None:
                first_actions = actions

            world, responses = world.step(actions)

            reward += responses.rewards * live.unsqueeze(-1).float()
            live = live & ~responses.terminal

        return reward, first_actions

    def __call__(self, world, value=True):
        envs = torch.arange(world.n_envs, device=world.device)
        totals = torch.stack([torch.zeros_like(world.valid, dtype=torch.float) for _ in range(world.n_seats)], -1)
        counts = torch.zeros_like(totals)
        for _ in range(self.n_rollouts):
            r, a = self.rollout(world)
            totals[envs, a[:, None], :] += r[:, None]
            counts[envs, a[:, None], :] += torch.ones_like(r[:, None])
        means = totals.div(counts).where(counts > 0, torch.zeros_like(counts))

        seat_means = means[envs, :, world.seats.long()]
        logits = torch.log_softmax(self.temperature*seat_means, -1)
        logits[~world.valid] = -np.inf

        return arrdict.arrdict(
            logits=logits,
            actions=torch.distributions.Categorical(logits=logits).sample(),
            v=totals.sum(-2).div(counts.sum(-2)))


def uniform_logits(valid):
    return torch.log(valid.float()/valid.sum(-1, keepdims=True))

class Win(arrdict.namedarrtuple(fields=('envs',))):
    """One-step one-seat win (+1)"""

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
    
        self.valid = torch.ones_like(self.envs[..., None].bool())
        self.seats = torch.zeros_like(self.envs)

        self.logits = uniform_logits(self.valid)
        self.v = torch.ones_like(self.valid.float().unsqueeze(-1))

    def step(self, actions):
        trans = arrdict.arrdict(
            terminal=torch.ones_like(self.envs.bool()),
            rewards=torch.ones_like(self.envs.float()))
        return self, trans

class WinnerLoser(arrdict.namedarrtuple(fields=('seats',))):
    """First seat wins each turn and gets +1; second loses and gets -1"""

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


class All(arrdict.namedarrtuple(fields=('history', 'count'))):
    """Players need to submit 1s each turn; if they do it every turn they get +1, else 0"""

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

        correct_so_far = (self.history == 1).sum(-2) == self.count[..., None]
        correct_to_go = 2**((self.history == 1).sum(-2) - self.length).float()

        v = correct_so_far.float()*correct_to_go
        self.v = v

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
    game = All.initial(n_envs=3, n_seats=2, length=3, device='cpu')
    for s in range(6):
        action = game.seats % 2
        game, transition = game.step(action.long())

    assert transition.terminal.all()
    assert (transition.rewards[:, 0] == 0).all()
    assert (transition.rewards[:, 1] == 1).all()


class SequentialMatrix(arrdict.namedarrtuple(fields=('payoffs', 'moves', 'seats'))):

    @classmethod
    def initial(cls, payoff, n_envs=1, device='cuda'):
        return cls(
            payoffs=torch.as_tensor(payoff).to(device)[None, ...].repeat(n_envs, 1, 1, 1),
            seats=torch.zeros((n_envs,), dtype=torch.int, device=device),
            moves=torch.full((n_envs, 2), -1, dtype=torch.int, device=device))

    @classmethod
    def dilemma(cls, *args, **kwargs):
        return cls.initial([
            [[0., 0.], [1., 0.]],
            [[0., 1.], [.5, .5]]], *args, **kwargs)

    @classmethod
    def antisymmetric(cls, *args, **kwargs):
        return cls.initial([
            [[1., 0.], [1., 1.]],
            [[0., 0.], [0., .1]]], *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.payoffs, torch.Tensor):
            return 

        self.n_envs = self.seats.size(-1)
        self.n_seats = 2
        self.device = self.seats.device

        self.obs_space = heads.Tensor((1,))
        self.action_space = heads.Masked(2)

        self.obs = self.moves[..., [0]].float()
        self.valid = torch.stack([torch.ones_like(self.seats, dtype=torch.bool)]*2, -1)

        self.envs = torch.arange(self.n_envs, device=self.device, dtype=torch.long)

    def step(self, actions):
        seats = self.seats + 1
        terminal = (seats == 2)

        moves = self.moves.clone()
        moves[self.envs, self.seats.long()] = actions.int()
        self._stats(moves[terminal])

        rewards = torch.zeros_like(self.payoffs[:, 0, 0])
        rewards[terminal] = self.payoffs[
            self.envs[terminal], 
            moves[terminal, 0].long(), 
            moves[terminal, 1].long()]

        seats[terminal] = 0
        moves[terminal] = -1

        world = type(self)(payoffs=self.payoffs, seats=seats, moves=moves)
        transitions = arrdict.arrdict(terminal=terminal, rewards=rewards)

        return world, transitions

    def _stats(self, moves):
        if not moves.nelement():
            return
        for i in range(2):
            for j in range(2):
                count = ((moves[..., 0] == i) & (moves[..., 1] == j)).sum()
                stats.mean(f'outcomes/{i}-{j}', count, moves.nelement()/2)