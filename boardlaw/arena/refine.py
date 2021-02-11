import time
from . import common, mohex
from logging import getLogger
from rebar import arrdict

log = getLogger(__name__)

def evaluate(run, idx, max_games=1024, target_std=.025):
    worlds = common.worlds(run, 2)
    agent = common.agent(run, idx)
    arena = mohex.Arena(worlds, max_games)

    start = time.time()
    trace = []
    while True:
        result, args = arena.play(agent)
        trace.append(result)
        if result.std < target_std:
            break
        if result.games >= max_games:
            break

        rate = (time.time() - start)/result.games
        log.info(f'{rate:.0f}s per game; {rate*result.games:.0f}s so far, {rate*max_games:.0f}s expected')

    return arrdict.stack(trace), args