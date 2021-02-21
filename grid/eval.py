import json
import hashlib
import aljpy
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

@aljpy.autocache('{key}')
def _parameters_cached(snaps, key):
    params = {}
    for idx, row in snaps.iterrows():
        s = storage.load_snapshot(row.run, row.idx)
        params[idx] = {**runs.info(row.run)['params'], 'samples': s['n_samples'], 'flops': s['n_flops']}
    return pd.DataFrame.from_dict(params, orient='index')

def parameters(snaps):
    key = hashlib.md5(json.dumps(snaps.index.tolist()).encode()).hexdigest()
    return _parameters_cached(snaps, key)

def compile(name):
    run, idx = name.split('.')
    log.info('Compiling...')
    agent = arena.common.agent(f'*{run}', int(idx), 'cuda')
    worlds = arena.common.worlds(f'*{run}', 2, 'cuda')
    
    decisions = agent(worlds)
    worlds.step(decisions.actions)
    log.info('Compiled')

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

def solve(games, wins, soln=None):
    try:
        return activelo.solve(games, wins, soln=soln)
    except InManifoldError:
        log.warning('Got a manifold error; throwing soln out')
        return None

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
        pn.theme(
            figure_size=(width, height), 
            strip_background=pn.element_rect(color='w', fill='w'),
            panel_grid=pn.element_line(color='k', alpha=.1))]

def poster_sizes():
    return pn.theme(text=pn.element_text(size=18),
                title=pn.element_text(size=18),
                legend_title=pn.element_text(size=18))

def plot_flops(snaps):
    return (pn.ggplot(snaps, pn.aes(x='flops', y='μ', group='run', color='factor(boardsize)'))
        + pn.geom_line()
        + pn.geom_point()
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_discrete(name='boardsize')
        + mpl_theme()
        + poster_sizes())

def plot_params(snaps):
    best = snaps.groupby(['boardsize', 'depth', 'width']).apply(lambda g: g.loc[g.μ.idxmax()])
    return (pn.ggplot(best, pn.aes(x='params', y='μ', color='factor(boardsize)'))
        + pn.geom_line()
        + pn.geom_point()
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_discrete(name='boardsize')
        + mpl_theme()
        + poster_sizes())


def load(boardsize, agents=None):
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

def run(boardsize=7, n_workers=8):
    snaps = snapshots(boardsize)
    snaps = pd.concat([snaps, parameters(snaps)], 1)
    snaps['nickname'] = snaps.run.str.extract('.* (.*)', expand=False) + '.' + snaps.idx.astype(str)
    snaps['params'] = params(snaps)
    snaps = snaps.set_index('nickname')

    games, wins = load(boardsize, snaps.index)

    compile(snaps.index[0])

    solver, soln, σ = None, None, None
    futures = {}
    with ProcessPoolExecutor(n_workers+1, initializer=init) as pool:
        while True:
            if solver is None:
                log.info('Submitting solve task')
                solver = pool.submit(solve, games, wins)
            elif solver.done():
                soln = solver.result()
                solver = None
                if soln is not None:
                    μ, σ = arena.analysis.difference(soln, soln.μ.idxmin())
                    log.info(f'μ_max: {μ.max():.1f}')
                    log.info(f'σ_ms: {σ.pow(2).mean()**.5:.2f}')

            for key, future in list(futures.items()):
                if future.done():
                    results = future.result()
                    games, wins = update(games, wins, results)
                    del futures[key]
                    save(boardsize, games, wins)
                    
                    log.info(f'saturation: {games.sum().sum()/N_ENVS/games.shape[0]:.0%}')

            while len(futures) < n_workers:
                if soln is None:
                    sugg = tuple(np.random.choice(games.index, (2,)))
                else:
                    sugg = suggest(soln)
                
                log.info('Submitting eval task')
                futures[sugg] = pool.submit(evaluate, *sugg)
        
            if σ is not None and σ.pow(2).mean()**.5 < .0:
                break
            
    snaps['μ'], snaps['σ'] = arena.analysis.difference(soln, soln.μ.idxmax())

@aljpy.autocache('{key}')
def _solve_cached(games, wins, key):
    return activelo.solve(games, wins)

def solve_cached(games, wins):
    gkey = hashlib.md5(games.to_json().encode()).hexdigest()
    wkey = hashlib.md5(wins.to_json().encode()).hexdigest()
    return _solve_cached(games, wins, gkey + wkey)

def vitals(boardsize):
    log.info(f'Generating vitals for {boardsize}')
    snaps = snapshots(boardsize)
    snaps = pd.concat([snaps, parameters(snaps)], 1)
    snaps['nickname'] = snaps.run.str.extract('.* (.*)', expand=False) + '.' + snaps.idx.astype(str)
    snaps['params'] = params(snaps)
    snaps = snaps.set_index('nickname')

    games, wins = load(boardsize, snaps.index)
    soln = solve_cached(games, wins)
    snaps['μ'], snaps['σ'] = arena.analysis.difference(soln, soln.μ.idxmax())

    return snaps

