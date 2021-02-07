from IPython import display
import matplotlib.pyplot as plt
import time
import jittens
import vast
import pandas as pd
from logging import getLogger
from pavlov import runs, stats
import ast

log = getLogger(__name__)

def acknowledged(desc):
    fresh = [j.params for j in jittens.jobs.jobs('fresh').values()]
    active = [j.params for j in jittens.jobs.jobs('active').values()]

    rs = runs.pandas().loc[lambda df: df.description == desc]
    fetched = [ast.literal_eval(r['JITTENS_PARAMS']) for _, r in rs._env.iteritems()]

    return fresh + active + fetched

def keystr(d):
    return str({k: d[k] for k in sorted(d)})

def is_missing(proposal, acks):
    return keystr(proposal) not in {keystr(a) for a in acks}

def launch():
    desc = 'main/5'
    acks = acknowledged(desc)
    for width in [1, 2, 4, 8]:
        for depth in [1, 2, 4, 8]:
            params = dict(width=width, depth=depth, boardsize=5, timelimit=45*60, desc=desc)
            if is_missing(params, acks):
                log.info(f'Launching {params}')
                jittens.jobs.submit(
                    cmd='python -c "from boardlaw.main import *; run_jittens()" >logs.txt 2>&1',
                    dir='.',
                    resources={'gpu': 1},
                    params=params)

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

def load_all():
    return load('main/', key=('boardsize', 'width', 'depth'))

def fetch():
    return jittens.manage.fetch('output/pavlov/', 'output/pavlov/')

def refresh():
    vast.jittenate(local=True)
    last_fetch = 0
    while not jittens.finished():
        display.clear_output(wait=True)
        jittens.refresh()
        time.sleep(15)
        
        if time.time() > last_fetch + 600:
            fetched = fetch()
            jittens.manage.cleanup(fetched)
            last_fetch = time.time()

def plot(desc, tail=5, ax=None):
    df = (load(desc)
            .tail(tail).mean()
            .rename('elo').reset_index()
            .pivot_table('elo', 'depth', 'width', aggfunc='max'))
    
    _, ax = plt.subplots() if ax is None else (None, ax)
    df.plot(title=desc, marker='.', cmap='viridis', grid=True, ax=ax)
    ax.set_xscale('log', basex=2)

    return ax
