import requests
import time
from boardlaw.arena import common, mohex, database
from logging import getLogger
from rebar import arrdict
from pavlov import runs

log = getLogger(__name__)

def rename(r, new):
    if isinstance(r, list):
        return [rename(rr, new) for rr in r]

    r = r.copy()
    r['names'] = [(new if n == 'agent' else n) for n in r['names']]
    return r

def assure(run, idx=None):
    if not runs.exists(run):
        p = runs.path(run)
        p.mkdir(exist_ok=True, parents=True)

        name = 'storage.latest.pkl' if idx is None else f'storage.snapshot.{idx}.pkl'


def evaluate(run, idx, max_games=1024, target_std=.025):
    """
    Memory usage:
        * 3b1w2d: 1.9G
        * 9b4096w1d: 2.5G
    """

    worlds = common.worlds(run, 2)
    agent = common.agent(run, idx)
    arena = mohex.CumulativeArena(worlds)

    name = 'latest' if idx is None else f'snapshot.{idx}'

    start = time.time()
    trace = []
    while True:
        soln, results = arena.play(agent)
        trace.append(soln)
        if soln.std < target_std:
            break
        if soln.games >= max_games:
            break

        rate = (time.time() - start)/(soln.games + 1e-6)
        log.info(f'{rate:.0f}s per game; {rate*soln.games:.0f}s so far, {rate*max_games:.0f}s expected')

        database.save(run, rename(results, name))

    return arrdict.stack(trace), results