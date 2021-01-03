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

class Responsibilities:

    def __init__(self, n_envs, n_agents, device):
        self.n_envs = n_envs
        self.counts = torch.full((n_envs, n_agents), np.nan, device=device)

    def update(self, selected):
        ones = self.counts.new_ones((self.n_envs, 1))
        self.counts.scatter_add_(1, selected[:, None], ones)

    def apportion(self, rewards):
        weights = self.counts/self.counts.sum(-1, keepdims=True)
        return rewards @ weights

    def terminate(self, terminal):
        self.counts[terminal] = 0

class Winrates:

    def __init__(self, n_agents, device):
        self.counts = torch.full((n_agents, n_agents), np.nan, device=device)

    def submit(self, apportioned):
        pass


class League:

    def __init__(self, n_envs, n_agents, device):
        self.step = 0

        self.resp = Responsibilities(n_envs, n_agents, device)
        self.wins = Winrates(n_agents, device)

    def submit(self, latest):
        pass

    def select(self, latest):
        pass

    def update(self, rewards, terminal):
        pass