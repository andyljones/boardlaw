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
from pathlib import Path

ROOT = Path('output/experiments/bee/eval')

log = getLogger(__name__)

set_start_method('spawn', True)

N_ENVS = 1024

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
    Brun, Bidx = Bname.split('.')
    A = arena.common.agent(f'*{Arun}', int(Aidx), 'cuda')
    B = arena.common.agent(f'*{Brun}', int(Bidx), 'cuda')
    worlds = arena.common.worlds(f'*{Arun}', N_ENVS, 'cuda')

    return arena.common.evaluate(worlds, [(Aname, A), (Bname, B)])

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
    print(pd.Timestamp.now('UTC').strftime('%H:%M:%S'))
    print(f'n rounds: {games.sum().sum()/N_ENVS}')
    print(f'saturation: {games.sum().sum()/N_ENVS/games.shape[0]:.0%}')
    print(f'coverage: {(soln.μ != 0).mean():.0%}')
    print(f'μ_max: {μ.max():.1f}')
    print(f'σ_ms: {σ.pow(2).mean()**.5:.2f}')
    print(f'n futures: {len(futures)}')

def suggest(soln):
    #TODO: Can I use the eigenvectors of the Σ to rapidly make orthogonal suggestions
    # for parallel exploration? Do I even need to go that complex - can I just collapse
    # Σ over the in-flight pairs?
    imp = activelo.improvement(soln)
    idx = np.random.choice(imp.stack().index, p=imp.values.flatten()/imp.sum().sum())
    return tuple(idx)

def params(df):
    intake = (df.boardsize**2 + 1)*df.width
    body = (df.width**2 + df.width) * df.depth
    output = df.boardsize**2 * (df.width + 1)
    return intake + body + output

def mpl_theme(width=12, height=8):
    return [
        pn.theme_matplotlib(),
        pn.guides(
            color=pn.guide_colorbar(ticks=False)),
        pn.theme(
            figure_size=(width, height), 
            strip_background=pn.element_rect(color='w', fill='w'),
            panel_grid=pn.element_line(color='k', alpha=.1))]

def poster_sizes():
    return pn.theme(text=pn.element_text(size=18),
                title=pn.element_text(size=18),
                legend_title=pn.element_text(size=18))

def plot(snaps):
    return (pn.ggplot(data=snaps)
        + pn.geom_line(pn.aes(x='flops', y='μ', group='run', color='params'))
        + pn.geom_point(pn.aes(x='flops', y='μ', group='run', color='params'))
        + pn.scale_x_continuous(trans='log10')
        + mpl_theme()
        + poster_sizes())

def load(boardsize, agents):
    path = ROOT / f'{boardsize}.json'
    if path.exists():
        entries = pd.read_json(path.open('r'))
        games = entries.pivot('agent', 'challenger', 'games').reindex(index=agents, columns=agents).fillna(0)
        wins = entries.pivot('agent', 'challenger', 'wins').reindex(index=agents, columns=agents).fillna(0)
    else:
        games  = pd.DataFrame(index=agents, columns=agents).fillna(0).astype(int)
        wins  = pd.DataFrame(index=agents, columns=agents).fillna(0).astype(int)

    return games, wins

def save(boardsize, games, wins):

    new = (pd.concat({
                    'games': games.stack(), 
                    'wins': wins.stack()}, 1)
                .loc[lambda df: df.games > 0]
                .rename_axis(index=('agent', 'challenger')))

    path = ROOT / f'{boardsize}.json'
    if path.exists():
        entries = pd.read_json(path.open('r')).set_index(['agent', 'challenger'])
        entries = new.combine_first(entries)
    else:
        entries = new

    path.parent.mkdir(exist_ok=True, parents=True)
    entries.reset_index().to_json(path)
    

def init():
    # Would be happier if I could get the index of the process in the pool so that 
    # exactly half the processes could get each GPU. But I don't seem to be able to!
    # Best I could manage would be to subclass ProcessPoolExecutor, and :effort:
    import os
    #TODO: Support variable number of GPUs
    device = os.getpid() % 2
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

def run(boardsize=3, n_workers=8):
    snaps = snapshots(boardsize)
    snaps = pd.concat([snaps, parameters(snaps)], 1)
    snaps['nickname'] = snaps.run.str.extract('.* (.*)', expand=False) + '.' + snaps.idx.astype(str)
    snaps['params'] = params(snaps)
    snaps = snaps.set_index('nickname')

    games, wins = load(boardsize, snaps.index)

    soln = None
    futures = {}
    solver = None
    with ProcessPoolExecutor(n_workers, initializer=init) as pool:
        while True:
            if solver is None:
                solver = pool.submit(activelo.solve, games, wins, soln=soln)
            elif solver.done():
                try:
                    soln = solver.result()
                except InManifoldError:
                    soln = None
                    log.warning('Got a manifold error; throwing soln out')
                finally:
                    solver = None

            for key, future in list(futures.items()):
                if future.done():
                    results = future.result()
                    games, wins = update(games, wins, results)
                    del futures[key]
                    save(boardsize, games, wins)

            while len(futures) < n_workers - 1:
                if soln is None:
                    sugg = tuple(np.random.choice(games.index, (2,)))
                else:
                    sugg = suggest(soln)
                
                futures[sugg] = pool.submit(evaluate, *sugg)
        
            if soln is not None:
                report(soln, games, futures)
                _, σ = arena.analysis.difference(soln, soln.μ.idxmin())
                if σ.pow(2).mean()**.5 < .01:
                    break
            
    snaps['μ'], snaps['σ'] = arena.analysis.difference(soln, soln.μ.idxmin())
