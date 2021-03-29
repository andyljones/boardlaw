import json
import torch
import pandas as pd
from .. import sql, mohex, analysis, hex
from . import common
from rebar import arrdict
from random import shuffle
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
from pkg_resources import resource_filename

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
    path = Path(resource_filename(__package__, 'data/mohex.json'))
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        worlds = initial_states()
        mhx = mohex.MoHexAgent()
        agents = [mhx, mhx]

        chunks = [list(range(i, i+n_agents)) for i in range(0, worlds.n_envs, n_agents)]
        shuffle(chunks)

        wins = torch.full((worlds.n_envs,), np.nan, device=worlds.device)
        for chunk in tqdm(chunks):
            wins[chunk] = evaluate(worlds[chunk], agents)

        path.write_text(json.dumps([int(w) for w in wins.cpu().int().numpy()]))
    
    return np.asarray(json.loads(path.read_text()), dtype=int)

def snapshot_wins(snap_id):
    row = sql.query('select * from snaps where id == ?', params=(snap_id,)).iloc[0]
    boardsize = sql.query('select boardsize from runs where run == ?', params=(row.run,)).iloc[0].boardsize
    worlds = initial_states(boardsize)

    agents = [
        common.agent(row.run, row.idx, device=worlds.device),
        mohex.MoHexAgent()]

    snap_wins = evaluate(worlds, agents)