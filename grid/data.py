import activelo
import pandas as pd
from pathlib import Path
from pavlov import runs, storage
import json
import aljpy
import hashlib
from logging import getLogger
from boardlaw import arena

ROOT = Path('output/experiments/bee/eval')

log = getLogger(__name__)

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


def params(df):
    intake = (df.boardsize**2 + 1)*df.width
    body = (df.width**2 + df.width) * df.depth
    output = df.boardsize**2 * (df.width + 1)
    return intake + body + output

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
        
@aljpy.autocache('{key}')
def _solve_cached(games, wins, key):
    return activelo.solve(games, wins)

def solve_cached(games, wins):
    gkey = hashlib.md5(games.to_json().encode()).hexdigest()
    wkey = hashlib.md5(wins.to_json().encode()).hexdigest()
    return _solve_cached(games, wins, gkey + wkey)

def snapshot_solns(boardsize=None, solve=True):
    if boardsize is None:
        return pd.concat([snapshot_solns(b) for b in range(3, 10)], 0)
    log.info(f'Generating vitals for {boardsize}')
    snaps = snapshots(boardsize)
    snaps = pd.concat([snaps, parameters(snaps)], 1)
    snaps['nickname'] = snaps.run.str.extract('.* (.*)', expand=False) + '.' + snaps.idx.astype(str)
    snaps['params'] = params(snaps)
    snaps = snaps.set_index('nickname')

    if solve:
        games, wins = load(boardsize, snaps.index)
        soln = solve_cached(games, wins)
        snaps['μ'], snaps['σ'] = arena.analysis.difference(soln, soln.μ.idxmax())

    return snaps

