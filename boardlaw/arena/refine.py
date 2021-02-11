import time
from . import common, mohex, database
from logging import getLogger
from rebar import arrdict

log = getLogger(__name__)

def evaluate(run, idx, max_games=1024, target_std=.025):
    worlds = common.worlds(run, 2)
    agent = common.agent(run, idx)
    arena = mohex.Arena(worlds, max_games)

    name = 'latest' if idx is None else f'snapshot.{idx}'

    start = time.time()
    trace = []
    while True:
        result = arena.play(agent, name=name)
        trace.append(result)
        if result.std < target_std:
            break
        if result.games >= max_games:
            break

        rate = (time.time() - start)/(result.games + 1e-6)
        log.info(f'{rate:.0f}s per game; {rate*result.games:.0f}s so far, {rate*max_games:.0f}s expected')

        database.save(run, result.results)