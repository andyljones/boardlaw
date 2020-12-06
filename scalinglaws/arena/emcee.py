import aljpy
import torch
import numpy as np
from rebar import arrdict
from logging import getLogger
import activelo

log = getLogger(__name__)

def permute_seats(seats, n_seats):
    offsets = torch.arange(len(seats), device=seats.device)
    return (seats + offsets) % n_seats

def unpermute_wins(wins, n_seats):
    unmixed = torch.full_like(wins, -1)
    envs = torch.arange(len(wins))
    for i in range(n_seats):
        seat_i = torch.full_like(envs, i)
        permuted = permute_seats(seat_i, n_seats)
        unmixed[envs, permuted] = wins[:, i]
    return unmixed.int()

class Emcee:

    def __init__(self, worldfunc, n_envs=1024, device='cpu'):
        self.worldfunc = worldfunc
        self.initial = worldfunc(n_envs, device=device)
        self.n_envs = n_envs
        self.device = device
        assert self.initial.n_seats == 2, 'Only support 2 seats for now'
        assert self.n_envs % np.math.factorial(self.initial.n_seats) == 0, 'Number of envs needs to be divisible by the number of permutations of seats'

        self.next_id = 0
        self.agents = {}
        self.names = {}

    def add_agent(self, name, agent):
        assert name not in self.agents
        self.names[self.next_id] = name
        self.agents[self.next_id] = agent
        self.next_id += 1

    def add_agents(self, agents):
        current = self.names.values()
        for name, agent in agents.items():
            if name not in current:
                self.add_agent(name, agent)

    def step(self, matchup):
        if len(self.agents) <= 1:
            return None

        worlds = self.initial.clone()
        terminal = torch.zeros((worlds.n_envs,), dtype=torch.bool, device=self.device)
        wins = torch.zeros((worlds.n_envs, worlds.n_seats), dtype=torch.int, device=self.device)
        while True:
            for i, id in enumerate(matchup):
                seats = permute_seats(worlds.seats, worlds.n_seats)
                mask = (seats == i) & ~terminal
                if mask.any():
                    decisions = self.agents[id](worlds[mask])
                    worlds[mask], transitions = worlds[mask].step(decisions.actions)
                    terminal[mask] = transitions.terminal
                    wins[mask] += (transitions.rewards == 1).int()

            if terminal.all():
                break
        
        wins = unpermute_wins(wins, worlds.n_seats).sum(0)
        result = aljpy.dotdict(
            names=[self.names[m] for m in matchup], 
            wins=wins, 
            games=int(terminal.sum()))

        return result

def test():
    from ..validation import WinnerLoser, RandomAgent

    def worldfunc(n_envs, device='cpu'):
        return WinnerLoser.initial(n_envs, device=device)

    mc = Emcee(worldfunc, n_envs=4)

    mc.add_agent('one', RandomAgent())
    mc.add_agent('two', RandomAgent())

    result = mc.step((0, 1))

    assert result.black_wins == 4

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