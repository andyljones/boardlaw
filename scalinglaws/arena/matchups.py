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

def sample(counts, n_samples):
    # Matchup ideals
    # * Shouldn't use more than n_cores MoHex agents
    # * Should concentrate PyTorch agents
    # * Should fill out some sort of pattern before filling out uniformly

    targets = counts.sum()*torch.full_like(counts, 1/counts.nelement())

    error = targets - counts
    error = error - error.min()
    prior = torch.ones_like(error)
    dist = error + prior/(error + prior).sum()
    
    n_agents = counts.size(1)
    sample = torch.distributions.Categorical(probs=dist.flatten()).sample((n_samples,))
    sample = torch.stack([sample // n_agents, sample % n_agents], -1)

    return sample

class AdaptiveMatcher:

    def __init__(self, worldfunc, n_envs=1, device='cpu'):
        self.worldfunc = worldfunc
        self.worlds = worldfunc(n_envs, device=device)
        self.device = device
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
        self.matchups = torch.randint(0, len(self.agents), (self.worlds.n_envs, self.worlds.n_seats), device=self.device)

    def _refresh(self, terminal):
        if not terminal.any():
            return 

        # Update the matchup distribution to better match the priorities
        scatter_add(self.counts, self.matchups[terminal])
        self.rewards[terminal] = 0

        # Add in-flight matchups to the counts
        counts = self.counts.clone()
        scatter_add(counts, self.matchups[~terminal])

        self.matchups[terminal] = sample(counts, terminal.sum())

    def step(self):
        if len(self.agents) <= 1:
            return []
        if self.matchups is None:
            self._initialize()

        hotseat = self.matchups.gather(1, self.worlds.seats[:, None])
        terminal = torch.zeros((self.worlds.n_envs,), dtype=torch.bool, device=self.device)
        for id in hotseat.unique():
            mask = hotseat == id
            decisions = self.agents[id](self.worlds[mask])
            self.worlds[mask], transitions = self.worlds[mask].step(decisions.actions)
            terminal[mask] = transitions.terminal
            self.rewards[mask] += transitions.rewards
        
        matchups = arrdict.numpyify(self.matchups[terminal])
        names = np.array(list(self.names.values()))[matchups]
        rewards = arrdict.numpyify(self.rewards[terminal])

        self._refresh(terminal)

        return [(tuple(n), tuple(r)) for n, r in zip(names, rewards)]

def test():
    from ..validation import All, RandomAgent

    def worldfunc(n_envs, device='cpu'):
        return All.initial(n_envs, 2, device=device)

    matcher = AdaptiveMatcher(worldfunc, n_envs=4)

    matcher.add_agent('one', RandomAgent())
    matcher.add_agent('two', RandomAgent())

    matcher.step()

def vectorization_benchmark(n_envs=None, T=10, device=None):
    import pandas as pd
    import aljpy
    from scalinglaws import worldfunc, agentfunc

    if device is None:
        df = pd.concat([vectorization_benchmark(n_envs, T, d) for d in ['cpu', 'cuda']], ignore_index=True)
        import seaborn as sns
        with sns.axes_style('whitegrid'):
            g = sns.FacetGrid(df, row='device', col='n_envs')
            g.map(sns.barplot, "n_agents", "rate")
        return df
    if n_envs is None:
        return pd.concat([vectorization_benchmark(n, T, device) for n in [60, 240, 960]], ignore_index=True)

    assert n_envs % 60 == 0

    results = []
    for n in [1, 2, 5, 10]:
        worlds = worldfunc(n_envs, device=device)
        agents = {i: agentfunc(device=device) for i in range(n)}

        envs = torch.arange(n_envs, device=device)/n_envs
        masks = {i: (i/n <= envs) & (envs < (i+1)/n) for i in agents}

        torch.cuda.synchronize()
        with aljpy.timer() as timer:
            for _ in range(T):
                for i, agent in agents.items():
                    decisions = agent(worlds[masks[i]])
                    worlds[masks[i]].step(decisions.actions)
            torch.cuda.synchronize()

        results.append({'n_agents': n, 'n_envs': n_envs, 'n_samples': T*n_envs, 'time': timer.time(), 'device': device})
        print(results[-1])

    df = pd.DataFrame(results)
    df['rate'] = (df.n_samples/df.time).astype(int)

    return df