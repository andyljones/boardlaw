import time
import plotnine as pn
import scipy as sp
import numpy as np
import activelo
import pandas as pd
from boardlaw import arena
from pavlov import storage, runs
from rebar import dotdict
from IPython import display
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method
from geotorch.exceptions import InManifoldError
from logging import getLogger

log = getLogger(__name__)

set_start_method('spawn', True)

N_ENVS = 512

def snapshots(boardsize):
    snapshots = {}
    for r in runs.runs(description=f'bee/{boardsize}'):
        for i, s in storage.snapshots(r).items():
            snapshots[r, i] = s
    return (pd.DataFrame.from_dict(snapshots, orient='index')
                    .rename_axis(index=('run', 'idx'))
                    .reset_index())

def parameters(snaps):
    params = {}
    for idx, row in snaps.iterrows():
        s = storage.load_snapshot(row.run, row.idx)
        params[idx] = {**runs.info(row.run)['params'], 'samples': s['n_samples'], 'flops': s['n_flops']}
    return pd.DataFrame.from_dict(params, orient='index')

def evaluate(Aname, Bname):

    Arun, Aidx = Aname.split('.')
    Brun, Bidx = Aname.split('.')
    A = arena.common.agent(f'*{Arun}', int(Aidx), 'cuda')
    B = arena.common.agent(f'*{Brun}', int(Bidx), 'cuda')
    worlds = arena.common.worlds(f'*{Arun}', N_ENVS, 'cuda')

    return arena.common.evaluate(worlds, {Aname: A, Bname: B})

def update(games, wins, results):
    games, wins = games.copy(), wins.copy()
    for result in results:
        games.loc[result.names[0], result.names[1]] += result.games
        games.loc[result.names[1], result.names[0]] += result.games
        wins.loc[result.names[0], result.names[1]] += result.wins[0]
        wins.loc[result.names[1], result.names[0]] += result.wins[1]
    return games, wins

def guess(games, wins, futures):
    mocks = []
    for f in futures:
        rate = (wins.at[f] + 1)/(games.at[f] + 2)
        ws = (int(rate*N_ENVS//2), N_ENVS//2 - int(rate*N_ENVS//2))
        mocks.append(dotdict.dotdict(
            names=f,
            wins=ws, 
            games=N_ENVS//2))
        mocks.append(dotdict.dotdict(
            names=f[::-1],
            wins=ws[::-1],
            games=N_ENVS//2))
    return update(games, wins, mocks)

def report(soln, games, futures):
    μ, σ = arena.analysis.difference(soln, soln.μ.idxmin())

    display.clear_output(wait=True)
    print(f'n rounds: {games.sum().sum()/N_ENVS}')
    print(f'saturation: {games.sum().sum()/N_ENVS/games.shape[0]:.0%}')
    print(f'coverage: {(soln.μ != 0).mean():.0%}')
    print(f'μ_max: {μ.max():.1f}')
    print(f'σ_ms: {σ.pow(2).mean()**.5:.2f}')
    print(f'n futures: {len(futures)}')

def suggest(soln, futures):
    imp = activelo.improvement(soln)
    return imp.stack().sort_values().tail((len(futures)+1)**2).sample(1).index[-1]

def params(df):
    intake = (df.boardsize**2 + 1)*df.width
    body = (df.width**2 + df.width) * df.depth
    output = df.boardsize**2 * (df.width + 1)
    return intake + body + output

def plot(snaps):
    (pn.ggplot(data=snaps)
        + pn.geom_line(pn.aes(x='flops', y='μ', group='run', color='params'))
        + pn.scale_x_continuous(trans='log10'))

def run(boardsize=3, n_workers=8):
    snaps = snapshots(boardsize)
    snaps = pd.concat([snaps, parameters(snaps)], 1)
    snaps['nickname'] = snaps.run.str.extract('.* (.*)', expand=False) + '.' + snaps.idx.astype(str)
    snaps['params'] = params(snaps)
    snaps = snaps.set_index('nickname')

    n_agents = len(snaps)
    wins  = pd.DataFrame(np.zeros((n_agents, n_agents)), snaps.index, snaps.index)
    games = pd.DataFrame(np.zeros((n_agents, n_agents)), snaps.index, snaps.index)

    soln = None
    futures = {}
    with ProcessPoolExecutor(n_workers) as pool:
        while True:
            if len(futures) < n_workers:
                try:
                    ggames, gwins = guess(games, wins, futures)
                    gsoln = activelo.solve(ggames, gwins, soln=soln)
                    sugg = activelo.suggest(gsoln)
                except InManifoldError:
                    soln = None
                    sugg = tuple(np.random.choice(snaps.index, (2,)))
                    log.warning('Got a manifold error; making a random suggestion')

                futures[sugg] = pool.submit(evaluate, *sugg)

            for key, future in list(futures.items()):
                if future.done():
                    results = future.result()
                    games, wins = update(games, wins, results)
                    del futures[key]
        
            soln = activelo.solve(games, wins, soln=soln)
            report(soln, games, futures)
            _, σ = arena.analysis.difference(soln, soln.μ.idxmin())
            if σ.pow(2).mean()**.5 < .1:
                break
            
    snaps['μ'], snaps['σ'] = arena.analysis.difference(soln, soln.μ.idxmin())
