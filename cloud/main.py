from .jittens.machines import forbid
import matplotlib.pyplot as plt
import time
from . import jittens
from cloud import vast
import pandas as pd
from logging import getLogger
from pavlov import runs, stats
import ast
from IPython import display

log = getLogger(__name__)

def acknowledged(desc):
    fresh = [j.params for j in jittens.jobs.jobs('fresh').values()]
    active = [j.params for j in jittens.jobs.jobs('active').values()]

    fetched = runs.pandas(description=desc)
    if fetched.size:
        fetched = fetched.params.values.tolist()
    else:
        fetched = []

    return fresh + active + fetched

def keystr(d):
    return str({k: d[k] for k in ('boardsize', 'width', 'depth')})

def is_missing(proposal, acks):
    return keystr(proposal) not in {keystr(a) for a in acks}

def launch():
    boardsize = 7
    desc = f'bee/{boardsize}'
    acks = acknowledged(desc)
    for nodes in [64]:
        for width in [256, 512]:
            for depth in [1, 2, 4]:
                params = dict(width=width, depth=depth, boardsize=boardsize, nodes=nodes, desc=desc)
                if is_missing(params, acks):
                    log.info(f'Launching {params}')
                    jittens.jobs.submit(
                        cmd='python -c "from boardlaw.main import *; run_jittens()" >logs.txt 2>&1',
                        dir='.',
                        resources={'gpu': 1},
                        params=params)

def fetch():
    return jittens.manage.fetch('output/pavlov/', 'output/pavlov/')

def refresh(forbidden=[]):
    vast.jittenate(local=True, ssh_accept=True, forbidden=forbidden)
    last_fetch = 0
    while not jittens.finished():
        try:
            display.clear_output(wait=True)
            jittens.refresh()
            time.sleep(15)

            if time.time() > last_fetch + 900:
                fetched = fetch()
                jittens.manage.cleanup(fetched)
                last_fetch = time.time()
        except Exception as e:
            log.info(f'Failed with error {e}')
            time.sleep(5)

    fetched = fetch()
    jittens.manage.cleanup(fetched)

def progress():
    active_jobs = jittens.jobs.jobs('active')
    active_runs = runs.pandas()._env.dropna().apply(lambda p: p.get('JITTENS_NAME', '') in active_jobs).pipe(lambda s: s.index[s])
    keys = runs.pandas().loc[active_runs, 'params'].apply(lambda p: (p['boardsize'], p['width'], p['depth']))
    return data.load_field('elo-mohex', 'μ').resample('1min').mean().bfill().notnull().sum().reindex(keys.values)

def offers():
    vast.offers('cuda_max_good >= 11.1 & gpu_name == "RTX 2080 Ti"')

def times():
    lines = jittens.manage.tails('output/pavlov/*/logs.0.txt', count=500, lineglob='*Approx*', display=False)

    times = {}
    js = jittens.jobs.jobs()
    ms = jittens.machines.machines()
    for j, ls in lines.items():
        m = ms[js[j].machine]
        port = getattr(m, 'connection_kwargs', {}).get('port', '')

        if ls:
            times[j] = (pd.to_timedelta(ls[-1].split(' ')[-2]), js[j].machine, port)
        else:
            times[j] = (pd.Timedelta('8h'), js[j].machine, port)
        
    return pd.DataFrame.from_dict(times, orient='index', columns=['time', 'machine', 'port'])