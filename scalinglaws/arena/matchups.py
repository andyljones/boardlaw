import aljpy
import torch
import numpy as np
from rebar import arrdict

def invert(d):
    return {v: k for k, v in d.items()}

def select(counts):
    # Matchup ideals
    # * Shouldn't use more than n_cores MoHex agents
    # * Should concentrate PyTorch agents
    # * Should fill out some sort of pattern before filling out uniformly

    targets = np.full_like(counts, 1/counts.nelement())
    error = targets - arrdict.numpyify(counts/counts.sum())
    return np.unravel_index(error.argmax(), error.shape)

class AdaptiveMatcher:

    def __init__(self, worldfunc, n_envs=1024, device='cpu'):
        self.worldfunc = worldfunc
        self.initial = worldfunc(n_envs, device=device)
        self.worlds = self.initial.clone()
        self.device = device
        assert self.worlds.n_seats == 2, 'Only support 2 seats for now'

        self.next_id = 0
        self.agents = {}
        self.names = {}
        self.matchup = (0,)*self.worlds.n_seats
        self.wins = torch.zeros((n_envs, self.worlds.n_seats), dtype=torch.int, device=device)

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

    def _refresh(self, terminal):
        if not terminal.any():
            return 

        # Update the matchup distribution to better match the priorities
        self.counts[tuple(self.matchup)] += terminal.sum()
        self.wins[terminal] = 0

        new_matchup = select(self.counts)
        if new_matchup != self.matchup:
            self.matchup = new_matchup
            self.worlds = self.initial.clone()
            self.wins[:] = 0

    def step(self):
        if len(self.agents) <= 1:
            return []

        seats = self.worlds.seats.clone()
        terminal = torch.zeros((self.worlds.n_envs,), dtype=torch.bool, device=self.device)
        for i, id in enumerate(self.matchup):
            mask = seats == id
            decisions = self.agents[id](self.worlds[mask])
            self.worlds[mask], transitions = self.worlds[mask].step(decisions.actions)
            terminal[mask] = transitions.terminal
            self.wins[mask] += (transitions.rewards == 1).long()
        
        if terminal.any():
            wins = tuple(map(int, self.wins[terminal].sum(0)))
        else: 
            wins = (0,)*self.worlds.n_seats
        result = aljpy.dotdict(
            black_name=self.names[self.matchup[0]], white_name=self.names[self.matchup[1]], 
            black_wins=wins[0], white_wins=wins[1],
            games=int(terminal.sum()))

        self._refresh(terminal)

        return result


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