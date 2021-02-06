import time
import jittens
import vast
import pandas as pd
from logging import getLogger
from pavlov import runs, stats
import ast

log = getLogger(__name__)

def launch():
    for width in [1, 2, 4, 8]:
        for depth in [1, 2, 4, 8]:
            params = dict(width=width, depth=depth, boardsize=3, timelimit=15*60)
            jittens.jobs.submit(
                cmd='python -c "from boardlaw.main import *; run_jittens()" >logs.txt 2>&1',
                dir='.',
                resources={'gpu': 1},
                params=params)


def run():
    vast.jittenate(local=True)
    launch()
    while not jittens.finished():
        jittens.manage()
        time.sleep(1)

def load(desc, key=('width', 'depth')):
    rs = runs.pandas().loc[lambda df: df.description.fillna('').str.startswith(desc)].index

    head, tail = [], []
    for r in rs:
        try:
            tail.append(stats.pandas(r, 'elo-mohex', 'μ'))
            d = ast.literal_eval(runs.info(r)['_env']['JITTENS_PARAMS'])
            head.append(tuple(d[f] for f in key))
        except Exception as e:
            log.info(f'Failed to load {r}: {e}')
            
    df = pd.DataFrame(tail, index=pd.MultiIndex.from_tuples(head)).T.sort_index(axis=1)
    df.columns.names = key

    return df

def plot(desc):
    df = (load(desc)
            .tail(5).mean()
            .rename('elo').reset_index()
            .pivot_table('elo', 'depth', 'width', aggfunc='max'))
    
    ax = df.plot(title=desc, marker='.', cmap='viridis', grid=True)
    ax.set_xscale('log', basex=2)