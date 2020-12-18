import torch
import numpy as np
from .. import mohex, hex
from . import database
from itertools import permutations, cycle, islice
from rebar import arrdict
from logging import getLogger

log = getLogger(__name__)

def run():
    agent = mohex.MoHexAgent()
    worlds = hex.Hex.initial(n_envs=8)

    universe = np.linspace(0, 1, 11)
    pairs = cycle(permutations(universe, 2))

    active = torch.tensor(list(islice(pairs, worlds.n_envs)))
    
    moves = torch.zeros((worlds.n_envs,))
    while True:
        agent.random = active.gather(1, worlds.seats[:, None].long().cpu())[:, 0]
        decisions = agent(worlds)
        worlds, transitions = worlds.step(decisions.actions)
        log.info('Stepped')

        moves += 1

        rewards = transitions.rewards.cpu()
        wins = (rewards == 1).int()
        terminal = transitions.terminal.cpu()
        for idx in terminal.nonzero(as_tuple=False).squeeze(-1):
            result = arrdict.arrdict(
                names=(f'mohex-{active[idx][0]:.2f}', f'mohex-{active[idx][1]:.2f}'),
                wins=tuple(map(int, wins[idx])),
                moves=int(moves[idx]),
                boardsize=worlds.boardsize)
            log.info(f'Storing {result.names[0]} v {result.names[1]}, {result.wins[0]}-{result.wins[1]} in {result.moves} moves')
            database.store('mohex', result)
            moves[idx] = 0
            active[idx] = torch.tensor(next(pairs))

        