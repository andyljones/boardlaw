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

def load(experiment):
    rs = runs.pandas().loc[lambda df: df.description == experiment].index

    ds = {}
    for r in rs:
        try:
            d = ast.literal_eval(runs.info(r)['_env']['JITTENS_PARAMS'])
            d['perf'] = stats.pandas(r, 'elo-mohex', 'μ').tail(5).mean()
            ds[r] = d
        except:
            log.info(f'Failed to load {r}')
            
    return pd.DataFrame.from_dict(ds, orient='index')