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

    def __init__(self, n_agents, device, frac=.2):
        self.n_games = torch.tensor(0, device=device)

        self.n_agents = n_agents
        self.step = 0
        self.stable = {}

        self.frac = frac
        self.active = None

    def record(self, trans):
        n_envs = trans.terminal.size(0)
        self.n_games += trans.terminal.sum()
        if not self.active and self.n_games > n_envs/self.frac and self.stable:
            self.n_games[()] = 0 
            self.active = np.random.choice(list(self.stable))
            log.info(f'League is active with a model {self.step - self.active} steps old')
        if self.active and self.n_games > n_envs/(1 - self.frac):
            self.n_games[()] = 0
            self.active = None
            log.info('League is inactive')
        self.step += 1

    def store(self, agent):
        if (len(self.stable) < self.n_agents) or (self.step % 1000 == 0):
            self.stable[self.step] = clone(agent.state_dict())
        while len(self.stable) > self.n_agents:
            del self.stable[min(self.stable)]

    def select(self, agent):
        if self.active and (self.step % 2 == 0):
            self.opponent.load_state_dict(self.stable[self.active])
            return self.opponent
        return agent


class Responsibilities:

    def __init__(self, n_envs, n_agents, device):
        self.n_envs = n_envs
        self.device = device
        # Fix the number of seats to 2 for now, keeps things simple
        self.counts = torch.full((n_envs, 2, n_agents), np.nan, device=device)

    def update(self, selected, seats):
        envs = torch.arange(self.n_envs, device=self.device)
        self.counts[envs, seats, selected] += 1

    def apportion(self, rewards):
        weights = self.counts/self.counts.sum(-1, keepdims=True)
        by_env = (rewards[..., None]*weights).sum(1)
        # This could plausibly be very large
        by_pair = (by_env[:, None, :]*by_env[:, :, None]).sum(0)
        return by_pair

    def terminate(self, terminal):
        self.counts[terminal] = 0

class Winrates:

    def __init__(self, n_agents, device):
        self.counts = torch.full((n_agents, n_agents), np.nan, device=device)

    def submit(self, apportioned):
        self.counts += apportioned

    def rates(self):
        return self.counts/self.counts.sum(-1, keepdims=True)

class League:

    def __init__(self, n_envs, n_agents, device):
        self.n_agents = n_agents
        self.stable = {}
        self.step = 0

        self.resp = Responsibilities(n_envs, n_agents, device)
        self.wins = Winrates(n_agents, device)

    def submit(self, latest):
        if len(s):
            pass

    def select(self, latest):


        self.selected = selected

    def update(self):
        self.resp.update()