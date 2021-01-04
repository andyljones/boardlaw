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

log = getLogger(__name__)

def clone(x):
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.clone().detach()
    else:
        return x

class SimpleLeague:

    def __init__(self, agent, n_opponents, n_stabled, device):
        self.n_games = torch.zeros((n_opponents,), device=device)

        self.step = 0
        self.stable = {i: agent.state_dict() for i in range(n_stabled)}

    def update(self, agent):
        agent.flip()
        if (len(self.stable) < self.n_agents) or (self.step % 1000 == 0):
            self.stable[self.step] = clone(agent.state_dict())
        while len(self.stable) > self.n_agents:
            del self.stable[min(self.stable)]

