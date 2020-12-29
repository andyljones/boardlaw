import pandas as pd
from pavlov import json

PREFIX = 'arena'

def save(run, result):
    if isinstance(result, list):
        for r in result:
            save(run, r)

    json.assure(run, PREFIX, {})
    with json.update(run, PREFIX) as d:
        key = tuple(result.names)
        if key not in d:
            d[key] = {'black_wins': 0, 'white_wins': 0, 'moves': 0}
        current = d[key]
        current['black_wins'] += result.wins[0]
        current['white_wins'] += result.wins[1]
        current['moves'] += result.moves

def pandas(run):
    return pd.DataFrame.from_dict(json.read(run, PREFIX))

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


