import plotnine as pn
import pandas as pd
from . import eval, plot, asymdata, data

def run(boardsize=9):
    snaps = data.snapshot_solns(9, solve=False)
    raw = asymdata.pandas(9).reset_index()
    raw['games'] = raw.black_wins + raw.white_wins

    regex = r'(?P<run>[\w-]+)\.(?P<idx>\d+)(?:\.(?P<nodes>\d+))?'
    black_spec = raw.black_name.str.extract(regex).rename(columns=lambda c: 'black_' + c).fillna(64)
    white_spec = raw.white_name.str.extract(regex).rename(columns=lambda c: 'white_' + c).fillna(64)
    raw = pd.concat([raw, black_spec, white_spec], 1)
    raw['black_nickname'] = raw.black_run + '.' + raw.black_idx
    raw['white_nickname'] = raw.white_run + '.' + raw.white_idx
    raw['black_nodes'] = raw.black_nodes.astype(float)
    raw['white_nodes'] = raw.white_nodes.astype(float)

    raw = raw.groupby(['black_nickname', 'black_nodes', 'white_nickname', 'white_nodes']).last().reset_index()
    raw['black_name'] = raw.black_nickname + '.' + raw.black_nodes.astype(int).astype(str)
    raw['white_name'] = raw.white_nickname + '.' + raw.white_nodes.astype(int).astype(str)    

    runs = raw.black_run.unique()
    subset = raw[raw.black_run.isin(runs) & raw.white_run.isin(runs)]

    df = subset.pivot('black_name', 'white_name', ['black_wins', 'white_wins'])
    wins = (df.black_wins + df.white_wins.T)
    games = wins + wins.T

    elos = asymdata.fast_elos(wins, games)
    elos = pd.Series(elos, wins.index)

    regex = r'(?P<run>[\w-]+)\.(?P<idx>\d+)(?:\.(?P<nodes>\d+))?'
    info = elos.index.str.extract(regex)
    info['nickname'] = info.run + '.' + info.idx
    info['nodes'] = info['nodes'].astype(float)
    info['elo'] = elos.values
    info = pd.merge(info[['nodes', 'nickname', 'elo']], snaps, left_on='nickname', right_on='nickname')    

    return (pn.ggplot(info)
        + pn.geom_point(pn.aes(x='flops', y='elo', color='nodes'))
        + pn.scale_x_continuous(trans='log10')
        + plot.mpl_theme())
