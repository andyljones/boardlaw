"""
* Get a stream of agents
* Get a stream of wins/losses
* Emit which agent to play in each step
* Use emissions and wins/losses to figure out each agent's 'responsibility' 
  for winning/losing a game.
* Use these responsibilities to figure out pairwise winrates
  * Convolve responsibilities with a blurring kernel, so that similar agents
    can have their winrates refined
  * Will need to extend responsibilities tracker into the future, or have 
    new agents inherit some fraction of the old agents' winrates
* Use pairwise winrates and previous emission to figure out who to emit next
  * Want to emit agents that'll win against the latest agent
  * Want to emit agents that're dissimilar to recently emitted agents
  * Want to bias strongly towards emitting the latest agent
"""
import numpy as np
import torch
from logging import getLogger
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

log = getLogger(__name__)

def clone(x):
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.clone().detach()
    else:
        return x

def assemble(agent, state_dict):
    new = deepcopy(agent)
    new.load_state_dict(state_dict)
    return new

class SimpleLeague:

    def __init__(self, evaluator, n_envs, n_opponents, n_stabled):
        self.n_envs = n_envs
        self.n_opponents = n_opponents

        [device] = [p.device for p in evaluator.parameters() if isinstance(p, torch.Tensor)]
        self.n_games = torch.zeros((n_opponents,), device=device)

        self.step = 0
        self.stable = {i: clone(evaluator.state_dict()) for i in range(n_stabled)}

        self.prime_frac = 3/4

    def init(self, evaluator):
        chunk = self.n_envs//4//self.n_opponents
        start = int(self.prime_frac*self.n_envs)
        assert start + chunk*self.n_opponents == self.n_envs

        idxs = np.random.choice(list(self.stable), (self.n_opponents,))
        evaluator.slices = [slice(start+i*chunk, start+(i+1)*chunk) for i, idx in enumerate(idxs)]
        evaluator.opponents = nn.ModuleList([assemble(evaluator, self.stable[idx]) for idx in idxs])

    def update(self, trans, evaluator):
        # Toggle the prime-only
        evaluator.prime_only = not evaluator.prime_only

        # Initialize
        if not evaluator.slices:
            self.init(evaluator)

        # Generate the prime mask
        is_prime = torch.full((self.n_envs,), True, device=trans.terminal.device)
        for s in evaluator.slices:
            is_prime[s] = False
            
        # Update the games count
        for i, s in enumerate(evaluator.slices):
            self.n_games[i] += trans[s].sum()

        # Swap out any over the limit
        replace = (self.n_games > self.n_envs).nonzero().squeeze(-1)
        for r in replace:
            new = np.random.choice(list(self.stable))
            evaluator.opponents[r] = assemble(evaluator, self.stable[new])

            self.n_games[r] = 0

        # Add to the stable
        if self.step % 1000 == 0:
            idx = np.random.choice(self.stable)
            self.stable[idx] = clone(evaluator.state_dict())

        return is_prime
