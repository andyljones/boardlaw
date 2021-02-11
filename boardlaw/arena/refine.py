import time
from . import common, mohex
from logging import getLogger

log = getLogger(__name__)

def evaluate(run, idx, max_games=1024, target_std=.05):
    worlds = common.worlds(run, 2)
    agent = common.agent(run, idx)
    arena = mohex.Arena(worlds, max_games)

    start = time.time()
    while True:
        result = arena.play(agent)
        if result.std < target_std:
            break
        if result.games >= max_games:
            break

        rate = (time.time() - start)/result.games
        log.info(f'{rate:.0f}s per game; {rate*result.games:.0f}s so far, {rate*max_games:0.f}s expected')

def test():
    import torch
    from boardlaw import mcts, hex
    print('in')

    hex.cuda.module()
    print('hex compiled')

    mcts.cuda.module()
    print('mcts compiled', flush=True)

    run = '2021-02-06 12-16-42 wan-ticks'
    worlds = common.worlds(run, 2)
    agent = common.agent(run, -1)
    print('ready')

    obs = worlds.obs
    print('done obs')

    worlds.step(torch.tensor([0, 0]))
    print('done step')

    m = mcts.MCTS(worlds, n_nodes=64)
    m.initialize(agent.network)
    print('initialized')

    m.descend()
    print('descend')

    m.simulate(agent.network)
    print('simulated')

    m.root()
    print('root')
