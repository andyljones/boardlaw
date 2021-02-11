import time
from collections import deque
import torch
import numpy as np
from .. import mohex, hex
from . import database, analysis, common
from rebar import arrdict
from pavlov import stats, runs, logs, storage
from logging import getLogger
import activelo
import pandas as pd
from functools import wraps
from contextlib import contextmanager
from multiprocessing import set_start_method, Process

log = getLogger(__name__)

BOARDSIZES = [3, 5, 7, 9, 11, 13]
RUN_NAMES = [f'mohex-{s}' for s in BOARDSIZES]

def elos(run_name, names=None, queue=[]):
    n = (database.symmetric_games(run_name)
            .reindex(index=names, columns=names)
            .fillna(0))
    w = (database.symmetric_wins(run_name)
            .reindex(index=names, columns=names)
            .fillna(0))
    
    for (i, j) in queue:
        ni, nj = names[i], names[j]
        w.loc[ni, nj] += (w.loc[ni, nj] + 1)/(n.loc[ni, nj] + 2)
        n.loc[ni, nj] += 1

    return activelo.solve(n.values, w.values)

def activelo_refill(run_name, names, queue, count=1):
    if len(queue) >= count:
        return 

    soln = elos(run_name, names, queue)
    imp = activelo.improvement(soln)
    while len(queue) < count:
        probs = imp.flatten()/imp.sum()
        idx = np.random.choice(np.arange(n.size), p=probs)
        pair = (idx // n.shape[0], idx % n.shape[0])

        log.info(f'Adding {pair} to the list')
        queue.append(pair)
        queue.append(pair[::-1])

def offdiag_refill(run, names, queue, count=1):
    n = (database.symmetric_games(run)
            .reindex(index=names, columns=names)
            .fillna(0))

    for (i, j) in queue:
        ni, nj = names[i], names[j]
        n.loc[ni, nj] += 1

    rs, cs = np.indices(n.shape)
    mask = ((rs == cs + 1) | (rs == cs - 1))
    excess = (n.values - n.values[mask].min())
    excess[~mask] = np.inf
    probs = np.exp(-excess)/np.exp(-excess).sum()
    while len(queue) < count:
        idx = np.random.choice(np.arange(n.size), p=probs.flatten())
        pair = (idx // n.shape[0], idx % n.shape[0])

        log.info(f'Adding {pair} to the list')
        queue.append(pair)
        queue.append(pair[::-1])

def accumulate(boardsize):
    """Run this to generate the `mohex-{boardsize}.json` files"""
    run_name = f'mohex-{boardsize}'
    agent = mohex.MoHexAgent()
    worlds = hex.Hex.initial(n_envs=8, boardsize=boardsize)

    universe = torch.linspace(0, 1, 11)
    names = sorted([f'mohex-{r}' for r in universe])

    queue = []
    offdiag_refill(run_name, names, queue, worlds.n_envs)

    active = torch.tensor(queue[:worlds.n_envs])
    queue = queue[worlds.n_envs:]

    moves = torch.zeros((worlds.n_envs,))
    while True:
        idxs = active.gather(1, worlds.seats[:, None].long().cpu())[:, 0]
        agent.random = universe[idxs]

        decisions = agent(worlds)
        worlds, transitions = worlds.step(decisions.actions)
        log.info('Stepped')

        moves += 1

        rewards = transitions.rewards.cpu()
        wins = (rewards == 1).int()
        terminal = transitions.terminal.cpu()
        for idx in terminal.nonzero(as_tuple=False).squeeze(-1):
            result = arrdict.arrdict(
                names=(f'mohex-{universe[active[idx][0]]:.2f}', f'mohex-{universe[active[idx][1]]:.2f}'),
                wins=tuple(map(int, wins[idx])),
                moves=int(moves[idx]),
                boardsize=worlds.boardsize)

            log.info(f'Storing {result.names[0]} v {result.names[1]}, {result.wins[0]}-{result.wins[1]} in {result.moves} moves')
            database.save(run_name, result)

            moves[idx] = 0

            offdiag_refill(run_name, names, queue)
            log.info(f'Starting on {queue[0]}')
            active[idx] = torch.tensor(queue[0])
            queue = queue[1:]

def append(df, name):
    names = list(df.index) + [name]
    return df.reindex(index=names, columns=names).fillna(0)

class Arena:

    def __init__(self, worlds, max_history):
        # Deferred import so the module can be imported from boardlaw
        self.worlds = worlds
        self.mohex = mohex.MoHexAgent()
        self.history = deque(maxlen=worlds.n_seats*max_history//self.worlds.n_envs)

    def play(self, agent):
        size = self.worlds.boardsize
        games = database.symmetric_games(f'mohex-{size}').pipe(append, 'agent')
        wins = database.symmetric_wins(f'mohex-{size}').pipe(append, 'agent')
        for result in self.history:
            games.loc[result.names[0], result.names[1]] += result.games
            games.loc[result.names[1], result.names[0]] += result.games
            wins.loc[result.names[0], result.names[1]] += result.wins[0]
            wins.loc[result.names[1], result.names[0]] += result.wins[1]

        soln = activelo.solve(games, wins)
        μ, σ = analysis.difference(soln, 'mohex-0.00', 'agent')
        log.info(f'Agent elo is {μ:.2f}±{σ:.2f} based on {int(games.loc["agent"].sum())} games')
        stats.mean_std('elo-mohex', μ, σ)

        imp = activelo.improvement(soln)
        imp = pd.DataFrame(imp, games.index, games.index)

        challenger = imp['agent'].idxmax()
        randomness = float(challenger.split('-')[1])
        self.mohex.random = randomness
        results = common.evaluate(self.worlds, {'agent': agent, challenger: self.mohex})
        log.info(f'Agent played {challenger}, {int(results[0].wins[0] + results[1].wins[1])}-{int(results[0].wins[1] + results[1].wins[0])}')
        self.history.extend(results)

        return arrdict.arrdict(games=games.loc['agent'].sum(), mean=μ, std=σ)

def run_sync(run):
    log.info('Arena launched')
    run = runs.resolve(run)

    log.info(f'Running arena for "{run}"')
    with logs.to_run(run), stats.to_run(run):
        worlds = common.worlds(run, 4)
        arena = Arena(worlds, 128)
        
        i = 0
        agent = None
        last_load, last_step = 0, 0
        while True:
            if time.time() - last_load > 15:
                last_load = time.time()
                agent = common.agent(run)
            
            if agent and (time.time() - last_step > 1):
                last_step = time.time()
                log.info('Running trial')
                arena.play(agent)
                i += 1

@wraps(run_sync)
@contextmanager
def run(*args, **kwargs):
    set_start_method('spawn', True)
    p = Process(target=run_sync, args=args, kwargs=kwargs, name='mohex-arena')
    try:
        p.start()
        yield p
    finally:
        for _ in range(50):
            if not p.is_alive():
                log.info('Arena monitor dead')
                break
            time.sleep(.1)
        else:
            log.info('Abruptly terminating arena monitor; it should have shut down naturally!')
            p.terminate()

