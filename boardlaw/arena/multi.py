import numpy as np
import torch
from itertools import combinations
from collections import deque

def scatter_inc_(totals, indices):
    assert indices.ndim == 2
    rows, cols = indices.T

    width = totals.shape[1]
    raveled = rows + width*cols

    ones = totals.new_ones((len(rows),))
    totals.view(-1).scatter_add_(0, raveled, ones)

def scatter_inc_symmetric_(totals, indices):
    scatter_inc_(totals, indices)
    scatter_inc_(totals, indices.flip(1))


class Tracker:

    def __init__(self, n_envs, n_envs_per, names):
        assert n_envs % n_envs_per == 0
        self.n_envs = n_envs
        self.n_envs_per = n_envs_per
        self.names = list(names)

        # Counts games that are either in-progress or that have been completed
        self.games = torch.zeros((len(names), len(names)), dtype=torch.int)

        self.live = torch.full((n_envs, 2), -1)

    def _live_counts(self):
        counts = torch.zeros_like(self.games)
        scatter_inc_symmetric_(counts, self.live[(self.live > -1).all(-1)])
        return counts

    def step(self, seats, terminal):
        # Kill off the finished games
        self.live[terminal] = -1

        # Figure out how the -1s in live should be repopulated
        while True:
            available = (self.live == -1).any(-1)
            remaining = (self.games < self.n_envs_per)
            if not (available.any() or remaining.any()):
                break

            counts = self._live_counts()
            goodness = (1 - 2*remaining.float())*(counts.sum(0, keepdim=True) * counts.sum(1, keepdim=True))

            choice = goodness.argmax()
            choice = (choice // len(self.names), choice % len(self.names))

            allocation = available.nonzero()[:self.n_envs_per]
            self.live[allocation] = torch.as_tensor(choice)

            self.games[choice] += len(allocation)

        # Suggest the most 'popular' agent  
        suggestion = self.live.gather(1, seats[:, None]).squeeze(-1)
        return suggestion

class MultiEvaluator:
    # Idea: keep lots and lots of envs in memory at once, play 
    # every agent against every agent simultaneously
    
    def __init__(self, worlds, agents):
        pass
    pass