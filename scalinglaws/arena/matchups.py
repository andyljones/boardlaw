import torch
import numpy as np
from rebar import arrdict

def invert(d):
    return {v: k for k, v in d.items()}

def scatter_add(counts, idxs):
    fst, snd = idxs[:, 0], idxs[:, 1]
    n_fst, n_snd = counts.shape
    ones = counts.new_ones((len(idxs),))
    counts.view(-1).scatter_add_(0, fst*n_snd + snd, ones)

class AdaptiveMatcher:

    def __init__(self, worldfunc, n_envs=1, device='cpu'):
        self.worldfunc = worldfunc
        self.worlds = worldfunc(n_envs, device=device)
        self.device = device
        self.seat = 0
        assert self.worlds.n_seats == 2, 'Only support 2 seats for now'

        self.next_id = 0
        self.agents = {}
        self.names = {}
        self.matchups = None
        self.rewards = torch.zeros((n_envs, self.worlds.n_seats), device=device)

        self.counts = torch.empty((0, 0))

    def add_agent(self, name, agent):
        assert name not in self.agents
        self.names[self.next_id] = name
        self.agents[self.next_id] = agent
        self.next_id += 1

        counts = torch.zeros((self.next_id, self.next_id), device=self.device)
        counts[:-1, :-1] = self.counts
        self.counts = counts

    def add_agents(self, agents):
        current = self.names.values()
        for name, agent in agents.items():
            if name not in current:
                self.add_agent(name, agent)

    def _initialize(self):
        self.matchups = torch.randint(0, len(self.agents), (self.worlds.n_envs, self.worlds.n_seats))

    def _refresh(self, terminal):
        # Update the matchup distribution to better match the priorities
        scatter_add(self.counts, self.matchups[terminal])
        self.rewards[terminal] = 0

        targets = self.counts.sum()*torch.full_like(self.counts, 1/self.counts.nelement())
        scatter_add(targets, self.matchups[~terminal])

        error = targets - self.counts
        error = error - error.min()
        prior = torch.ones_like(error)
        dist = error + prior/(error + prior).sum()
        
        n_agents = self.counts.size(1)
        sample = torch.distributions.Categorical(probs=dist.flatten()).sample((terminal.sum(),))
        sample = torch.stack([sample // n_agents, sample % n_agents], -1)

        self.matchups[terminal] = sample 

    def step(self):
        if len(self.agents) == 0:
            return []
        if self.matchups is None:
            self._initialize()

        terminal = torch.zeros((len(self.matchups)), dtype=torch.bool, device=self.device)
        for (i, id) in enumerate(self.agents):
            mask = (self.matchups[:, self.seat] == i) & (self.worlds.seats == self.seat)
            if mask.any():
                decisions = self.agents[id](self.worlds[mask])
                self.worlds[mask], transitions = self.worlds[mask].step(decisions.actions)
                terminal[mask] = transitions.terminal
                self.rewards[mask] += transitions.rewards

        self.seat = (self.seat + 1) % self.worlds.n_seats
        
        matchups = arrdict.numpyify(self.matchups[terminal])
        names = np.array(list(self.names.keys()))[matchups]
        rewards = arrdict.numpyify(self.rewards[terminal])

        if terminal.any():
            self._refresh(terminal)

        return [(tuple(n), tuple(r)) for n, r in zip(names, rewards)]

def test():
    from ..validation import All, RandomAgent

    def worldfunc(n_envs, device='cpu'):
        return All.initial(n_envs, 2, device=device)

    matcher = AdaptiveMatcher(worldfunc, n_envs=4)

    matcher.add_agent('one', RandomAgent())
    matcher.add_agent('two', RandomAgent())