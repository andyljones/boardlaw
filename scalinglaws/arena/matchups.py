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

class AdaptiveMatcher:

    def __init__(self, worldfunc, device='cpu'):
        self.worldfunc = worldfunc
        self.worlds = worldfunc(0)
        self.device = device

        self.next_id = 0
        self.agents = {}
        self.names = {}
        self.matchups = torch.empty((0, self.worlds.n_seats), dtype=torch.long, device=device)
        self.rewards = torch.zeros((), device=device)

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

    def agent_indices(self):
        idxs = np.arange(self.n_envs)
        fstidxs, sndidxs, _ = np.unravel_index(idxs, (self.n_agents, self.n_agents, self.n_copies))

        self.worlds = self.worldfunc(len(idxs), self.device)
        self.idxs = torch.as_tensor(np.stack([fstidxs, sndidxs], -1), device=self.device) 

        self.rewards = torch.zeros((self.n_envs, self.worlds.n_seats))

    def _refresh(self, terminal):
        # Update the matchup distribution to better match the priorities
        pass

    def step(self):
        terminal=torch.zeros((len(self.matchups)), dtype=torch.bool, device=self.device)
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

        self.refresh(terminal)

        return [(tuple(n), tuple(r)) for n, r in zip(names, rewards)]