import numpy as np
import activelo
import pandas as pd
from pathlib import Path
from pavlov import runs, storage
import ast
import json
import aljpy
import hashlib
from logging import getLogger
from boardlaw import arena

ROOT = Path('output/experiments/bee/eval')

log = getLogger(__name__)

def runs_pandas(*args, **kwargs):
    df = runs.pandas(*args, **kwargs)
    df['params'] = df.params.map(lambda x: {} if isinstance(x, float) else x)
    df['description'] = df.description.fillna('')
    return df

def snapshots(boardsize):
    #TODO: I need a better indexing system than 'bee' and 'cat', jesus
    rows = (runs_pandas()
                .loc[lambda df: df.description.str.match('^(bee|cat)/')]
                .loc[lambda df: df.params.apply(lambda d: d.get('boardsize') == boardsize)])

    snapshots = {}
    for r, _ in rows.iterrows():
        for i, s in storage.snapshots(r).items():
            snapshots[r, i] = s

    snapshots = (pd.DataFrame.from_dict(snapshots, orient='index')
                    .rename_axis(index=('run', 'idx'))
                    .reset_index())
    
    return pd.merge(snapshots, rows[['description', '_env']], on='run')

@aljpy.autocache('{key}')
def _parameters_cached(snaps, key):
    params = {}
    for idx, row in snaps.iterrows():
        s = storage.load_snapshot(row.run, row.idx)
        # Gotta combine two sources of param data here cause I forgot to save down the params for the cat/nodes runs
        env_params = ast.literal_eval(row['_env']['JITTENS_PARAMS'])
        saved_params = runs.info(row.run)['params']

        params[idx] = {
            **env_params, 
            **saved_params,
            'samples': s['n_samples'], 
            'flops': s['n_flops']}
    return pd.DataFrame.from_dict(params, orient='index')

def parameters(snaps):
    key = hashlib.md5(json.dumps(snaps.index.tolist()).encode()).hexdigest()
    return _parameters_cached(snaps, key)

def params(df):
    intake = (df.boardsize**2 + 1)*df.width
    body = (df.width**2 + df.width) * df.depth
    output = df.boardsize**2 * (df.width + 1)
    return intake + body + output

def load(boardsize=None, agents=None):
    if boardsize is None:
        games, wins = zip(*[load(b) for b in range(3, 10)])
        return pd.concat(games), pd.concat(wins)
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
        
@aljpy.autocache('{key}')
def _solve_cached(games, wins, key):
    return activelo.solve(games, wins)

def solve_cached(games, wins):
    gkey = hashlib.md5(games.to_json().encode()).hexdigest()
    wkey = hashlib.md5(wins.to_json().encode()).hexdigest()
    return _solve_cached(games, wins, gkey + wkey)

def snapshot_solns(boardsize=None, solve=True):
    if boardsize is None:
        return pd.concat([snapshot_solns(b, solve) for b in range(3, 10)], 0)
    log.info(f'Generating vitals for {boardsize}')
    snaps = snapshots(boardsize)
    snaps = pd.concat([snaps, parameters(snaps)], 1)
    snaps['nickname'] = snaps.run.str.extract('.* (.*)', expand=False) + '.' + snaps.idx.astype(str)
    snaps['params'] = params(snaps)
    snaps = snaps.set_index('nickname')
    assert snaps.index.to_series().value_counts().eq(1).all()

    if solve:
        games, wins = load(boardsize, snaps.index)
        soln = solve_cached(games, wins)
        snaps['μ'], snaps['σ'] = arena.analysis.difference(soln, soln.μ.idxmax())

    return snaps
