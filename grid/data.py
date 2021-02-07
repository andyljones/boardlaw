from pavlov import runs, stats
import ast
import pandas as pd
from logging import getLogger

log = getLogger(__name__)

TAILS = {3: 5, 5: 15, 7: 30}

def load_field(*args, key=('boardsize', 'width', 'depth')):
    rs = runs.pandas().loc[lambda df: df.description.fillna('').str.startswith('main/')].index

    head, tail = [], []
    for r in rs:
        try:
            tail.append(stats.pandas(r, *args))
            d = ast.literal_eval(runs.info(r)['_env']['JITTENS_PARAMS'])
            head.append(tuple(d[f] for f in key))
        except Exception as e:
            log.info(f'Failed to load {r}: {e}')
            
    df = pd.DataFrame(tail, index=pd.MultiIndex.from_tuples(head)).T.sort_index(axis=1)
    df.columns.names = key

    return df.mean(axis=1, level=[0, 1, 2])

def load():
    return pd.concat({
        'elo': load_field('elo-mohex', 'μ'),
        'samples': load_field('count.samples')}, 1)

def tail_means(df):
    tails = pd.concat({b: df[b].dropna(0, 'all').tail(t).mean().mean(level=[0, 1]) for b, t in TAILS.items()})
    tails.index.names = ['boardsize', 'width', 'depth']
    return tails.rename('elo').reset_index()

def min_elos():
    # Values from running the code below
    return pd.Series({3: -3.09, 5: -6.34, 7: -9.03, 9: -12.64, 11: -16.21})
    from boardlaw.arena import mohex
    return {b: mohex.elos(f'mohex-{b}').μd[-1, 0].round(2) for b in [3, 5, 7, 9, 11]}

def augmented():
    df = load()
    df = tail_means(df)
    df['state'] = df.depth*df.width
    df['params'] = df.depth*df.width**2
    df['flops'] = df.depth*df.width**3
    df['rel_elo'] = 1 - df.elo / min_elos().reindex(df.boardsize.values).values
    return df
