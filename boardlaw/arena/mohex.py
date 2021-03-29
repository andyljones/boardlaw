import torch
import pandas as pd
from .. import sql, mohex, analysis, hex
from . import common
from rebar import arrdict
from random import shuffle
from tqdm.auto import tqdm
import numpy as np

def evaluate(snap_id, n_envs=2):
    row = sql.query('select * from snaps where id == ?', params=(snap_id,)).iloc[0]
    worlds = common.worlds(row.run, n_envs=n_envs)
    ags = {
        snap_id: common.agent(row.run, row.idx, device='cpu'),
        'mhx': mohex.MoHexAgent()}
    return common.evaluate(worlds, ags) 

def tree_sample(boardsize=7, size=1024, seed=1):
    state = torch.get_rng_state()
    torch.manual_seed(seed)

    worlds = hex.Hex.initial(size, boardsize, device='cuda')
    terminal = [torch.zeros(size, device=worlds.device, dtype=torch.bool)]
    trace  = [worlds]
    while not terminal[-1].all():
        actions = torch.distributions.Categorical(worlds.valid.float()).sample()
        worlds, transitions = worlds.step(actions)
        trace.append(worlds)
        terminal.append(terminal[-1] | transitions.terminal)
    trace = arrdict.stack(trace)
    terminal = arrdict.stack(terminal)

    max_rows = (~terminal).cumsum(0).max(0).values
    rows = torch.rand(size, device=worlds.device).mul(max_rows).int()
    rows = torch.minimum(rows, max_rows)

    cols = torch.arange(size, device=worlds.device)
    sample = trace[rows, cols]

    torch.set_rng_state(state)
    return sample
    
def initial_states(boardsize=7):
    count = boardsize**4
    first = torch.arange(count, device='cuda') // boardsize**2
    second = torch.arange(count, device='cuda') % boardsize**2

    factored = torch.stack([first // boardsize, first % boardsize], -1)
    transposed = factored[:, 1]*boardsize + factored[:, 0]
    mask = transposed != second

    worlds = hex.Hex.initial(mask.sum(), boardsize, device=mask.device)
    worlds, _ = worlds.step(first[mask])
    worlds, _ = worlds.step(second[mask])

    return worlds

def evaluate(worlds, agents):
    worlds = worlds.copy()
    terminal = torch.full((worlds.n_envs,), False, device=worlds.device)
    rewards = torch.full((worlds.n_envs, worlds.n_seats), 0., device=worlds.device)
    while not terminal.all():
        [idx] = worlds[~terminal].seats.unique()
        decisions = agents[idx](worlds[~terminal])
        worlds[~terminal], transitions = worlds[~terminal].step(decisions.actions)
        
        rewards[~terminal] += transitions.rewards
        terminal[~terminal] = transitions.terminal
    return (rewards == 1).float().argmax(-1).float()

def reference_wins(n_agents=8):
    worlds = initial_states()
    mhx = mohex.MoHexAgent()
    agents = [mhx, mhx]

    chunks = [list(range(i, i+n_agents)) for i in range(0, worlds.n_envs, n_agents)]
    shuffle(chunks)

    wins = torch.full((worlds.n_envs,), np.nan, device=worlds.device)
    for chunk in tqdm(chunks):
        wins[chunk] = evaluate(worlds[chunk], agents)
    
    return wins