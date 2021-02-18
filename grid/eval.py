import numpy as np
import activelo
import pandas as pd
from boardlaw import arena
from pavlov import storage, runs
from rebar import dotdict
from IPython import display

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

def update(games, wins, results):
    games, wins = games.copy(), wins.copy()
    for result in results:
        games.loc[result.names[0], result.names[1]] += result.games
        games.loc[result.names[1], result.names[0]] += result.games
        wins.loc[result.names[0], result.names[1]] += result.wins[0]
        wins.loc[result.names[1], result.names[0]] += result.wins[1]
    return games, wins

def report(soln):
    μ, σ = arena.analysis.difference(soln, soln.μ.idxmin())

    display.clear_output(wait=True)
    print(f'nonzero: {(soln.μ != 0).mean():.0%}')
    print(f'μ_max: {μ.max():.1f}')
    print(f'σ_ms: {σ.pow(2).mean()**.5:.2f}')

def run(boardsize=3):
    worlds, agents = worlds_and_agents(boardsize)

    n_agents = len(agents)
    wins  = pd.DataFrame(np.zeros((n_agents, n_agents)), list(agents), list(agents))
    games = pd.DataFrame(np.zeros((n_agents, n_agents)), list(agents), list(agents))

    soln = None
    while True:
        soln = activelo.solve(games, wins, soln=soln)
        sugg = activelo.suggest(soln)

        results = arena.common.evaluate(worlds, {n: agents[n] for n in sugg})
        games, wins = update(games, wins, results)

        report(soln)
