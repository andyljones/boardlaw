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

def assemble(agentfunc, state_dict):
    new = agentfunc().evaluator.prime
    new.load_state_dict(state_dict)
    return new

class SimpleLeague:

    def __init__(self, agentfunc, evaluator, n_envs, n_opponents=4, n_stabled=16, prime_frac=3/4):
        self.n_envs = n_envs
        self.n_opponents = n_opponents

        [device] = {p.device for p in evaluator.parameters() if isinstance(p, torch.Tensor)}
        self.n_games = torch.zeros((n_opponents,), device=device)

        self.step = 0
        self.stable = {i: evaluator.state_dict() for i in range(n_stabled)}

        prime_frac = 3/4
        self.n_prime_envs = int(prime_frac*self.n_envs)
        self.n_oppo_envs = int((1 - prime_frac)*self.n_envs//self.n_opponents)
        assert self.n_prime_envs + self.n_oppo_envs*self.n_opponents == self.n_envs

        idxs = np.random.choice(list(self.stable), (self.n_opponents,))
        start, chunk = self.n_prime_envs, self.n_oppo_envs
        evaluator.slices = [slice(start + i*chunk, start + (i+1)*chunk) for i, idx in enumerate(idxs)]
        evaluator.opponents = nn.ModuleList([assemble(agentfunc, self.stable[idx]) for idx in idxs])

    def update(self, evaluator, transition):
        # Toggle the prime-only
        evaluator.prime_only = not evaluator.prime_only

        # Generate the prime mask
        is_prime = torch.full((self.n_envs,), True, device=transition.terminal.device)
        for s in evaluator.slices:
            is_prime[s] = False
            
        # Update the games count
        for i, s in enumerate(evaluator.slices):
            self.n_games[i] += transition.terminal[s].sum()

        # Figure out who's been playing too long. Stagger it a bit so they don't all change at once
        threshold = torch.linspace(1.5, 2.5, self.n_opponents, device=transition.terminal.device).mul(self.n_oppo_envs).int()
        (replace,) = (self.n_games >= threshold).nonzero(as_tuple=True)
        # Swap out any over the limit
        for i in replace:
            new = np.random.choice(list(self.stable))
            evaluator.opponents[i].load_state_dict(self.stable[new])
            log.info(f'New opponent #{new} is {self.step - new} steps old')

            self.n_games[i] = 0

        # Add to the stable
        if self.step % 60 == 0:
            old = np.random.choice(list(self.stable))
            del self.stable[old]
            self.stable[self.step] = clone(evaluator.state_dict())
            log.info(f'Network #{self.step} stabled; #{old} removed')

        self.step += 1

        return is_prime
