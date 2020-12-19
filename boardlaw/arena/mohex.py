import torch
import numpy as np
from .. import mohex, hex
from . import database
from itertools import permutations, cycle, islice
from rebar import arrdict
from logging import getLogger
import activelo

log = getLogger(__name__)

def refill(run_name, names, queue, count=1):
    if len(queue) >= count:
        return 

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

    soln = activelo.solve(n.values, w.values)
    imp = activelo.improvement(soln, 1)
    while len(queue) < count:
        probs = imp.flatten()/imp.sum()
        idx = np.random.choice(np.arange(n.size), p=probs)
        pair = (idx // n.shape[0], idx % n.shape[0])

        log.info(f'Adding {pair} to the list')
        queue.append(pair)
        queue.append(pair[::-1])

def run(boardsize):
    run_name = f'mohex-{boardsize}'
    agent = mohex.MoHexAgent()
    worlds = hex.Hex.initial(n_envs=8, boardsize=boardsize)

    universe = torch.linspace(0, 1, 11)
    names = sorted([f'mohex-{r}' for r in universe])

    queue = []
    refill(run_name, names, queue, worlds.n_envs)

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
            database.store(run_name, result)

            moves[idx] = 0

            refill(run_name, names, queue)
            log.info(f'Starting on {queue[0]}')
            active[idx] = torch.tensor(queue[0])
            queue = queue[1:]