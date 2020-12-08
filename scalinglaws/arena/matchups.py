import aljpy
import torch
import numpy as np
from rebar import dotdict
from logging import getLogger
import activelo
from itertools import permutations

log = getLogger(__name__)

def matchup_patterns(n_seats):
    return torch.as_tensor(list(permutations(range(n_seats))))

def matchup_indices(n_envs, n_seats):
    patterns = matchup_patterns(n_seats)
    return patterns.repeat((n_envs//len(patterns), 1))

def gather(wins, matchup_idxs, names):
    names = np.array(names)
    n_envs, n_seats = matchup_idxs.shape
    results = []
    for p in matchup_patterns(n_seats):
        ws = wins[(matchup_idxs == p).all(-1)].sum(0) 
        results.append(dotdict.dotdict(
            names=tuple(names[p]),
            wins=tuple(map(float, ws)),
            games=float(ws.sum())))
    return results

def evaluate(worlds, agents):
    assert worlds.n_seats == 2, 'Only support 2 seats for now'
    assert worlds.n_envs % np.math.factorial(worlds.n_seats) == 0, 'Number of envs needs to be divisible by the number of permutations of seats'
    assert len(agents) == worlds.n_seats, 'Need to pass one agent per seat'

    envs = torch.arange(worlds.n_envs, device=worlds.device)
    terminal = torch.zeros((worlds.n_envs,), dtype=torch.bool, device=worlds.device)
    wins = torch.zeros((worlds.n_envs, worlds.n_seats), dtype=torch.int, device=worlds.device)
    matchup_idxs = matchup_indices(worlds.n_envs, worlds.n_seats).to(worlds.device)
    while True:
        for i, id in enumerate(agents):
            mask = (matchup_idxs[envs, worlds.seats.long()] == i) & ~terminal
            if mask.any():
                decisions = agents[id](worlds[mask])
                worlds[mask], transitions = worlds[mask].step(decisions.actions)
                terminal[mask] = transitions.terminal
                wins[mask] += (transitions.rewards == 1).int()

        if terminal.all():
            break
    
    results = gather(wins, matchup_idxs, agents)
    return results

def test():
    from ..validation import WinnerLoser, RandomAgent

    worlds = WinnerLoser.initial(4, device='cpu')
    results = evaluate(worlds, {'one': RandomAgent(), 'two': RandomAgent()})

    assert results[0].black_wins == 2
    assert results[1].black_wins == 2