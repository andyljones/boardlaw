import time
import torch
import numpy as np
from rebar import dotdict
from logging import getLogger
from itertools import permutations
from pavlov import storage, runs
from ..mcts import MCTSAgent
from ..hex import Hex

log = getLogger(__name__)

def agent(run, idx=None, device='cpu'):
    try:
        network = storage.load_raw(run, 'model', device)
    except IOError:
        log.warn(f'No model file for "{run}"')
        return None

    agent = MCTSAgent(network)

    try:
        if idx is None:
            sd = storage.load_latest(run)
        else:
            sd = storage.load_snapshot(run, idx)
    except IOError:
        log.warn(f'No state dict file for "{run}"')
        return None

    agent.load_state_dict(sd['agent'])

    return agent

def worlds(run, n_envs, device='cpu'):
    boardsize = runs.info(run)['params']['boardsize']
    return Hex.initial(n_envs, boardsize, device)

def matchup_patterns(n_seats):
    return torch.as_tensor(list(permutations(range(n_seats))))

def matchup_indices(n_envs, n_seats):
    patterns = matchup_patterns(n_seats)
    return patterns.repeat((n_envs//len(patterns), 1))

def gather(wins, moves, times, matchup_idxs, names, boardsize):
    names = np.array(names)
    n_envs, n_seats = matchup_idxs.shape
    results = []
    for p in matchup_patterns(n_seats):
        ws = wins[(matchup_idxs == p).all(-1)].sum(0) 
        ms = moves[(matchup_idxs == p).all(-1)].sum(0) 
        ts = times[(matchup_idxs == p).all(-1)].sum(0) 
        results.append(dotdict.dotdict(
            names=tuple(names[p]),
            wins=tuple(map(float, ws)),
            moves=float(ms),
            games=float(ws.sum()),
            times=float(ts),
            boardsize=boardsize))
    return results

def evaluate(worlds, agents):
    assert worlds.n_seats == 2, 'Only support 2 seats for now'
    assert worlds.n_envs % np.math.factorial(worlds.n_seats) == 0, 'Number of envs needs to be divisible by the number of permutations of seats'
    assert len(agents) == worlds.n_seats, 'Need to pass one agent per seat'

    envs = torch.arange(worlds.n_envs, device=worlds.device)
    terminal = torch.zeros((worlds.n_envs,), dtype=torch.bool, device=worlds.device)
    wins = torch.zeros((worlds.n_envs, worlds.n_seats), dtype=torch.int, device=worlds.device)
    moves = torch.zeros((worlds.n_envs,), dtype=torch.int, device=worlds.device)
    times = torch.zeros((worlds.n_envs,), dtype=torch.float, device=worlds.device)
    matchup_idxs = matchup_indices(worlds.n_envs, worlds.n_seats).to(worlds.device)
    while True:
        for i, id in enumerate(agents):
            mask = (matchup_idxs[envs, worlds.seats.long()] == i) & ~terminal
            if mask.any():
                start = time.time()
                decisions = agents[id](worlds[mask], eval=True)
                worlds[mask], transitions = worlds[mask].step(decisions.actions)
                terminal[mask] = transitions.terminal
                end = time.time()

                wins[mask] += (transitions.rewards == 1).int()
                moves[mask] += 1
                times[mask] += (end - start)/mask.sum()

        if terminal.all():
            break
    
    results = gather(wins.cpu(), moves.cpu(), times.cpu(), matchup_idxs.cpu(), list(agents), worlds.boardsize)
    return results

def test_evaluate():
    from boardlaw.validation import WinnerLoser, RandomAgent

    worlds = WinnerLoser.initial(4, device='cpu')
    results = evaluate(worlds, {'one': RandomAgent(), 'two': RandomAgent()})

    assert results[0].wins == (2., 0.)
    assert results[1].wins == (2., 0.)