import json
import torch
import pandas as pd
from .. import sql, elos, mohex, analysis, hex
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
    worlds = worlds.clone()
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

def calibrate(agent_id, mhx=None, n_envs=128):
    row = sql.query('select * from agents_details where id == ?', params=(agent_id,)).iloc[0]

    worlds = hex.Hex.initial(n_envs, row.boardsize)

    ag = common.agent(row.run, row.idx, device=worlds.device)
    ag.kwargs['n_nodes'] = row.test_nodes

    mhx = mohex.MoHexAgent() if mhx is None else mhx
    agents = {
        agent_id: ag,
        None: mhx}
    results = common.evaluate(worlds, agents)
    sql.save_mohex_trials(results)

def run(boardsize):
    ags = sql.agent_query().query('test_nodes == 64 & boardsize == 7')

    trials = sql.trial_query(boardsize, 'bee/%')
    trials = trials[trials.black_agent.isin(ags.index) & trials.white_agent.isin(ags.index)]
    ws, gs = elos.symmetrize(trials)
    ags['elo'] = elos.solve(ws, gs)
    targets = ags.query('elo > -1').index

    extant = sql.mohex_trial_query(boardsize)
    extant = (set(extant.white_agent.dropna().astype(int).values) | 
              set(extant.black_agent.dropna().astype(int).values))

    choices = set(targets) - extant

    mhx = mohex.MoHexAgent()
    for c in tqdm(choices):
        calibrate(c, mhx=mhx)

def analyze(boardsize):
    ags = sql.agent_query().loc[lambda df: df.boardsize == boardsize].query('test_nodes == 64 & boardsize == 7')
    trials = sql.trial_query(boardsize, 'bee/%')
    trials = trials[trials.black_agent.isin(ags.index) & trials.white_agent.isin(ags.index)]
    ws, gs = elos.symmetrize(trials)
    ags['elo'] = elos.solve(ws, gs)

    mhx = sql.mohex_trial_query(boardsize)
    black_wins = mhx[['black_agent', 'black_wins', 'white_wins']].dropna().set_index('black_agent')[['black_wins', 'white_wins']]
    white_wins = mhx[['white_agent', 'black_wins', 'white_wins']].dropna().set_index('white_agent')[['black_wins', 'white_wins']]
    black_wins = black_wins.groupby(black_wins.index).sum()
    white_wins = white_wins.groupby(white_wins.index).sum()

    rate = (black_wins.black_wins + white_wins.white_wins)/(black_wins.sum(1) + white_wins.sum(1))
    rate.index = rate.index.astype(int)

    mhx_elo = np.log(rate) - np.log(1 - rate)

    pd.concat({'old': ags.elo, 'new': mhx_elo}, 1).dropna().corr()