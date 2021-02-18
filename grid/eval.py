import numpy as np
import activelo
import pandas as pd
from boardlaw import arena
from pavlov import storage, runs
from rebar import dotdict

def snapshots(boardsize):
    snapshots = {}
    for r in runs.runs(description=f'bee/{boardsize}'):
        for i, s in storage.snapshots(r).items():
            snapshots[r, i] = s
    return (pd.DataFrame.from_dict(snapshots, orient='index')
                    .rename_axis(index=('run', 'idx'))
                    .reset_index())

def worlds_and_agents(boardsize):
    snaps = snapshots(boardsize)
    worlds = arena.common.worlds(snaps.run[0], 512, 'cuda')

    agents = {}
    for _, row in snaps.iterrows():
        name = f'{row.run.split(" ")[-1]}.{row.idx}'
        agents[name] = arena.common.agent(row.run, row.idx, 'cuda')
    
    return worlds, agents

def run(boardsize=3):
    worlds, agents = worlds_and_agents(boardsize)

    n_agents = len(agents)
    wins  = pd.DataFrame(np.zeros((n_agents, n_agents)), list(agents), list(agents))
    games = pd.DataFrame(np.zeros((n_agents, n_agents)), list(agents), list(agents))

    soln = activelo.solve(games, wins)
    sugg = activelo.suggest(soln)

    results = arena.common.evaluate(worlds, {n: agents[n] for n in sugg})
    pass
    # for runs 

