import torch
import numpy as np
from rebar import arrdict

class Matcher:

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