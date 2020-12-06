from scalinglaws.learning import gather
import aljpy
import torch
import numpy as np
from rebar import arrdict
from logging import getLogger
import activelo

log = getLogger(__name__)

def matchup_indices(n_envs, n_seats):
    #TODO: Generalise this to more than 2 seats
    assert n_seats == 2
    offsets = torch.arange(n_envs)
    return torch.stack([(i + offsets) % n_seats for i in range(n_seats)], -1)

def gather_wins(wins, names):
    n_seats = wins.shape[-1]
    # (env, matchup component, seat)
    gathered = torch.full(wins.shape + (n_seats,), -1, device=wins.device, dtype=wins.dtype)
    envs = torch.arange(len(wins))
    for i in range(n_seats):
        seat_i = torch.full_like(envs, i)
        matchup_idxs = matchup_indices(seat_i, n_seats)
        gathered[envs, matchup_idxs] = wins
    return gathered.sum(0)

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
        envs = torch.arange(worlds.n_envs, device=worlds.device)
        terminal = torch.zeros((worlds.n_envs, worlds.n_seats), dtype=torch.bool, device=self.device)
        wins = torch.zeros((worlds.n_envs, worlds.n_seats, worlds.n_seats), dtype=torch.int, device=self.device)
        matchup_idxs = matchup_indices(self.n_envs, worlds.n_seats)
        while True:
            for i, id in enumerate(matchup):
                mask = (matchup_idxs[envs, worlds.seats] == i) & ~terminal
                if mask.any():
                    decisions = self.agents[id](worlds[mask])
                    worlds[mask], transitions = worlds[mask].step(decisions.actions)
                    terminal[mask] = transitions.terminal
                    wins[mask] += (transitions.rewards == 1).int()

            if terminal.all():
                break
        
        wins = gather_wins(wins)
        names = [self.names[m] for m in matchup]
        result = aljpy.dotdict(
            names=, 
            wins=list(wins), 
            games=worlds.n_seats*int(terminal.sum()))

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