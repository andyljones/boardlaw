import torch
import numpy as np
from rebar import arrdict

class SimpleMatcher:

    def __init__(self, worldfunc, agents, device='cpu', n_copies=1):
        self.worldfunc = worldfunc
        self.device = device
        self.agents = {k: agent.to(device) for k, agent in agents.items()}

        self.n_agents = len(self.agents)
        self.n_envs = n_copies*self.n_agents**2
        self.n_copies = n_copies

        self.worlds = None
        self.idxs = None
        self.seat = 0

        self.rewards = None

        self.initialize()

    def initialize(self):
        idxs = np.arange(self.n_envs)
        fstidxs, sndidxs, _ = np.unravel_index(idxs, (self.n_agents, self.n_agents, self.n_copies))

        self.worlds = self.worldfunc(len(idxs), self.device)
        self.idxs = torch.as_tensor(np.stack([fstidxs, sndidxs], -1), device=self.device) 

        self.rewards = torch.zeros((self.n_envs, self.worlds.n_seats))

    def step(self):
        terminal=torch.zeros((self.n_envs), dtype=torch.bool, device=self.device)
        for (i, name) in enumerate(self.agents):
            mask = (self.idxs[:, self.seat] == i) & (self.worlds.seats == self.seat)
            if mask.any():
                decisions = self.agents[name](self.worlds[mask])
                self.worlds[mask], transitions = self.worlds[mask].step(decisions.actions)
                terminal[mask] = transitions.terminal
                self.rewards[mask] += transitions.rewards

        self.seat = (self.seat + 1) % self.worlds.n_seats
        
        idxs = arrdict.numpyify(self.idxs)
        names = np.array(list(self.agents.keys()))[idxs[terminal]]
        rewards = arrdict.numpyify(self.rewards[terminal])

        self.rewards[terminal] = 0.

        return [(tuple(n), tuple(r)) for n, r in zip(names, rewards)]

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

        self.next_id = 0
        self.agents = {}
        self.names = {}
        self.matchups = torch.empty((0, self.worlds.n_seats), dtype=torch.long, device=device)
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

    def drop_agent(self, name):
        id = invert(self.names)[name]
        terminal = self.matchups == self.names[id]
        del self.agents[id]
        del self.names[id]
        self._refresh(terminal)

    def _refresh(self, terminal):
        # Update the matchup distribution to better match the priorities
        if terminal.any():
            if len(self.matchups) == 0:
                targets = self.counts
            else:
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

            if len(self.matchups) == 0:
                self.matchups = sample
            else:
                self.matchups[terminal] = sample 

    def step(self):
        if len(self.matchups) == 0:
            terminal = torch.ones((self.worlds.n_envs,), dtype=torch.bool, device=self.device)
            self._refresh(terminal)

        terminal = torch.zeros((len(self.matchups)), dtype=torch.bool, device=self.device)
        for (i, id) in enumerate(self.agents):
            mask = (self.matchups[:, self.seat] == i) & (self.worlds.seats == self.seat)
            if mask.any():
                decisions = self.agents[id](self.worlds[mask])
                self.worlds[mask], transitions = self.worlds[mask].step(decisions.actions)
                terminal[mask] = transitions.terminal
                self.rewards[mask] += transitions.rewards

        self.seat = (self.seat + 1) % self.worlds.n_seats
        
        matchups = arrdict.numpyify(self.matchups)
        names = np.array(list(self.agents.keys()))[matchups[terminal]]
        rewards = arrdict.numpyify(self.rewards[terminal])

        self._refresh(terminal)

        return [(tuple(n), tuple(r)) for n, r in zip(names, rewards)]

def test():
    from ..validation import All, RandomAgent

    def worldfunc(*args, **kwargs):
        return All.initial(*args, **kwargs, n_seats=2)
        
    matcher = AdaptiveMatcher(worldfunc, n_envs=4)

    matcher.add_agent('one', RandomAgent())
    matcher.add_agent('two', RandomAgent())

    results = []
    for _ in range(4):
        results.append(matcher.step())
