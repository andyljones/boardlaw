from contextlib import contextmanager
import pandas as pd
from pkg_resources import resource_filename
from pavlov import json, runs
from pathlib import Path
import json as json_

PREFIX = 'arena'

KEYS = ['black_name', 'white_name']

def _to_dict(l):
    # This indirection is because we can't store multi-part keys in a JSON. Ugh.
    return {tuple(r[n] for n in KEYS): {k: v for k, v in r.items() if k not in KEYS} for r in l}

def _to_list(d):
    return [{**dict(zip(KEYS, k)), **v} for k, v in d.items()]

def boardsize_path(boardsize):
    return Path(f'output/experiments/eval/asym/{boardsize}.json')

def assure(boardsize):
    path = boardsize_path(boardsize)
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text('[]')

@contextmanager
def update(boardsize):
    p = boardsize_path(boardsize)
    contents = json_.loads(p.read_text())
    yield contents
    p.write_text(json_.dumps(contents))

def save(results):
    if not results:
        return
    
    boardsize = results[0].boardsize
    
    assure(boardsize)
    with update(boardsize) as l:
        d = _to_dict(l)
        for result in results:
            k = tuple(result.names)
            if k not in d:
                d[k] = {'black_wins': 0, 'white_wins': 0, 'moves': 0, 'times': 0.}
            v = d[k]
            v['black_wins'] += result.wins[0]
            v['white_wins'] += result.wins[1]
            v['moves'] += result.moves
            v['times'] += result.times

        l[:] = _to_list(d)

def pandas(boardsize):
    path = boardsize_path(boardsize)
    if path.exists():
        contents = json_.loads(path.read_text())
    else:
        contents = []

    if contents:
        return pd.DataFrame(contents).set_index(KEYS)
    else:
        return pd.DataFrame(columns=['black_name', 'white_name', 'black_wins', 'white_wins', 'moves']).set_index(KEYS)

def pandas_elos(boardsize):
    from . import data
    import activelo

    snaps = data.snapshot_solns(5, solve=False)

    raw = pandas(5)
    raw['games'] = raw.black_wins + raw.white_wins

    games = raw.games.unstack().reindex(index=snaps.index, columns=snaps.index).fillna(0)

    black_wins = raw.black_wins.unstack().reindex_like(games)
    white_wins = raw.white_wins.unstack().reindex_like(games).T

    ws = (black_wins/games + white_wins/games.T)/2*(games + games.T)/2.
    gs = (games + games.T)/2.

    soln = activelo.solve(gs.fillna(0), ws.fillna(0))

    snaps['μ'] = soln.μ - soln.μ.max()

    return snaps