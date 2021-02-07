from plotnine import *
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
    return str({k: d[k] for k in ('boardsize', 'width', 'depth')})

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

def load_board(b, key=('width', 'depth')):
    rs = runs.pandas().loc[lambda df: df.description == f'main/{b}'].index

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

def load_full():
    return load_board('main/', key=('boardsize', 'width', 'depth'))

def tail_means(df):
    tails = {3: 5, 5: 15, 7: 30}
    tails = pd.concat({b: df[b].dropna(0, 'all').tail(t).mean().mean(level=[0, 1]) for b, t in tails.items()})
    tails.index.names = ['boardsize', 'width', 'depth']
    return tails.rename('elo').reset_index()


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
    df = (load_board(desc)
            .tail(tail).mean()
            .rename('elo').reset_index()
            .pivot_table('elo', 'depth', 'width', aggfunc='max'))
    
    _, ax = plt.subplots() if ax is None else (None, ax)
    df.plot(title=desc, marker='.', cmap='viridis', grid=True, ax=ax)
    ax.set_xscale('log', basex=2)

    return ax

def min_elos():
    # Values from running the code below
    return pd.Series({3: -3.09, 5: -6.34, 7: -9.03, 9: -12.64, 11: -16.21})
    from boardlaw.arena import mohex
    return {b: mohex.elos(f'mohex-{b}').μd[-1, 0].round(2) for b in [3, 5, 7, 9, 11]}

def plot_sigmoids(full):
    data = tail_means(full)
    data['state'] = data.depth*data.width
    data['params'] = data.depth*data.width**2
    data['flops'] = data.depth*data.width**3

    data['rel_elo'] = 1 - data.elo / min_elos().reindex(data.boardsize.values).values
    (ggplot(data=data)
        + geom_point(mapping=aes(x='width', y='rel_elo', color='depth'))
        + facet_wrap('boardsize', nrow=1)
        + scale_x_continuous(trans='log2')
        + scale_color_continuous(trans='log2')
        + theme_matplotlib()
        + theme(
            figure_size=(18, 6), 
            strip_background=element_rect(color='w', fill='w'),
            panel_grid=element_line(color='k', alpha=.1)))