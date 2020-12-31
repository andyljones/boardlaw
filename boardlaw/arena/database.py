import pandas as pd
from pavlov import json
from pathlib import Path
import json as json_

PREFIX = 'arena'

KEYS = ['black_name', 'white_name']

def _to_dict(l):
    # This indirection is because we can't store multi-part keys in a JSON. Ugh.
    return {tuple(r[n] for n in KEYS): {k: v for k, v in r.items() if k not in KEYS} for r in l}

def _to_list(d):
    return [{**dict(zip(KEYS, k)), **v} for k, v in d.items()]

def save(run, result):
    if isinstance(result, list):
        for r in result:
            save(run, r)
        return

    json.assure(run, PREFIX, [])
    with json.update(run, PREFIX) as l:
        d = _to_dict(l)
        k = tuple(result.names)
        if k not in d:
            d[k] = {'black_wins': 0, 'white_wins': 0, 'moves': 0}
        v = d[k]
        v['black_wins'] += result.wins[0]
        v['white_wins'] += result.wins[1]
        v['moves'] += result.moves

        l[:] = _to_list(d)

def pandas(run):
    # Need to come up with a better way of handling 'special' runs
    if run.startswith('mohex'):
        contents = json_.loads(Path(f'output/arena/{run}.json').read_text())
    else:
        contents = json.read(run, PREFIX, [])
    if contents:
        return pd.DataFrame(contents).set_index(KEYS)
    else:
        return pd.DataFrame(columns=['black_name', 'white_name', 'black_wins', 'white_wins', 'moves']).set_index(KEYS)

def summary(run):
    raw = pandas(run)
    if len(raw) == 0:
        columns = pd.MultiIndex.from_product([['black_wins', 'white_wins',], []])
        return pd.DataFrame(columns=columns)
    df = (raw
            .groupby(['black_name', 'white_name'])
            [['black_wins', 'white_wins', 'moves']]
            .sum()
            .unstack())
    
    names = sorted(list(set(df.index) | set(df.columns.get_level_values(1))))
    df = df.reindex(index=names).reindex(columns=names, level=1)
    return df.fillna(0)

def games(run):
    df = summary(run)
    if len(df) == 0:
        df = pd.DataFrame()
        df.index.name = 'black_name'
        df.columns.name = 'white_name'
        return df
    return df.white_wins + df.black_wins

def wins(run, min_games=-1):
    df = summary(run)
    if len(df) == 0:
        return pd.DataFrame()
    return df.black_wins

def moves(run):
    df = summary(run)
    if len(df) == 0:
        return pd.DataFrame()
    return df.moves

def symmetric_games(run):
    g = games(run)
    return g + g.T

def symmetric_wins(run, min_games=-1):
    games = symmetric_games(run)
    df = summary(run)
    if len(df) == 0:
        return pd.DataFrame()
    return (df.black_wins + df.white_wins.T).where(games > min_games)

def symmetric_moves(run):
    m = moves(run)
    return m + m.T

def symmetric(run, agents=None):
    games = symmetric_games(run)
    wins = symmetric_wins(run)
    if agents is not None:
        agents = list(agents)
        games = games.reindex(index=agents, columns=agents).fillna(0)
        wins = wins.reindex(index=agents, columns=agents).fillna(0)
    return games, wins
