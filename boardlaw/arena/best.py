import pandas as pd
from .. import sql, elos
from . import mohex, common
import numpy as np
from tqdm.auto import tqdm

MIDS = {
    3: 112,
    4: 1127,
    5: 3109,
    6: 4332,
    7: 7497,
    8: 10775,
    9: 14652}

TOPS = {
    3: 121, 
    4: 922, 
    5: 2994, 
    6: 4024, 
    7: 23047, 
    8: 10605, 
    9: 14576}

def frontier_participants(ags, boardsize):
    from analysis import data
    ags = ags.query('test_nodes == 64').loc[lambda df: df.boardsize == boardsize]
    ys = data.interp_curves(ags)

    selection = []
    for flops, r in ys.iterrows():
        run = r.idxmax()
        snaps = ags.loc[ags.run == run].sort_values('train_flops')
        dists = snaps.train_flops.pipe(np.log10) - np.log10(flops)
        if (dists == 0).any():
            selection.append((dists == 0).idxmax())
        else:
            if (dists < 0).any():
                selection.append(dists[dists < 0].index[-1])
            if (dists > 0).any():
                selection.append(dists[dists > 0].index[0])
    return list(set(selection))

def best_v_mohex(boardsize):
    return mohex.calibrations(boardsize).sort_values('winrate').agent_id.iloc[-1]

def available(ref_id, n_envs):
    seen = sql.query('''
        select black_agent, white_agent from trials 
        where 
            (black_agent == ? or white_agent == ?) and
            (black_wins + white_wins) >= ?''', params=(int(ref_id), int(ref_id), n_envs//2))
    seen = set(seen.black_agent) | set(seen.white_agent)

    boardsize = sql.query('select boardsize from agents_details where id == ?', params=(ref_id,)).iloc[0].boardsize
    return (sql.agent_query()
                .query('test_nodes == 64')
                .loc[lambda df: df.boardsize == boardsize]
                .drop(seen, 0, errors='ignore')
                .index)

def evaluate(ref_id, n_envs=64*1024):
    total = len(available(ref_id, n_envs))

    with tqdm(total=total, desc=str(ref_id)) as pbar:
        while True:
            av = available(ref_id, n_envs)
            if len(av) == 0:
                break
            agent_id = np.random.choice(av)

            agents = {
                agent_id: common.sql_agent(agent_id, device='cuda'),
                ref_id: common.sql_agent(ref_id, device='cuda')}

            worlds = common.sql_world(ref_id, n_envs, device='cuda')
            results = common.evaluate(worlds, agents)

            sql.save_trials(results)

            target = total - len(av)
            pbar.update(target - pbar.n)

def best_rates(ref_id):
    from . import mohex

    black = sql.query('select * from trials where black_agent == ?', params=(ref_id,))
    white = sql.query('select * from trials where white_agent == ?', params=(ref_id,))

    black = black.set_index('white_agent').assign(games=lambda df: df.black_wins + df.white_wins).rename(columns={'white_wins': 'wins'})[['wins', 'games']]
    white = white.set_index('black_agent').assign(games=lambda df: df.black_wins + df.white_wins).rename(columns={'black_wins': 'wins'})[['wins', 'games']]

    trials = (black + white)
    trials = trials.groupby(trials.index).sum()

    return pd.concat({
        'best_games': trials.games,
        'best_elo': np.log(trials.wins + 1) - np.log(trials.games - trials.wins + 1)}, 1)


