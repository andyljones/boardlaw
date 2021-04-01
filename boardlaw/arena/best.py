import pandas as pd
from .. import sql, elos
from . import mohex, common
import numpy as np
from tqdm.auto import tqdm

MIDS = {9: 14652}

def frontier_participants(boardsize):
    from analysis import data
    ags = data.load().loc[lambda df: df.boardsize == boardsize]
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

def _available(boardsize, best_id, n_envs):
    seen = sql.query('''
        select black_agent, white_agent from trials 
        where 
            (black_agent == ? or white_agent == ?) and
            (black_wins + white_wins) >= n_envs''', params=(int(best_id), int(best_id), n_envs))
    seen = set(seen.black_agent) | set(seen.white_agent)

    return (sql.agent_query()
                .query('test_nodes == 64')
                .loc[lambda df: df.boardsize == boardsize]
                .drop(seen, 0, errors='ignore')
                .index)

def evaluate_best(boardsize, n_envs=64*1024):
    if boardsize is None:
        for b in range(3, 10):
            evaluate_best(b)
    from . import mohex

    best_id = mohex.calibrations(boardsize).sort_values('winrate').agent_id.iloc[-1]
    total = len(_available(boardsize, best_id, n_envs))

    with tqdm(desc=f'{boardsize}', total=total) as pbar:
        while True:
            av = _available(boardsize, best_id, n_envs)
            if len(av) == 0:
                break
            agent_id = np.random.choice(av)

            agents = {
                agent_id: common.sql_agent(agent_id, device='cuda'),
                best_id: common.sql_agent(best_id, device='cuda')}

            worlds = common.sql_world(agent_id, n_envs, device='cuda')
            results = common.evaluate(worlds, agents)

            sql.save_trials(results)

            target = total - len(av)
            pbar.update(target - pbar.n)

def best_rates(boardsize):
    from . import mohex

    best_id = mohex.best_agent(boardsize)

    black = sql.query('select * from trials where black_agent == ?', params=(best_id,))
    white = sql.query('select * from trials where white_agent == ?', params=(best_id,))

    black = black.set_index('white_agent').assign(games=lambda df: df.black_wins + df.white_wins).rename(columns={'white_wins': 'wins'})[['wins', 'games']]
    white = white.set_index('black_agent').assign(games=lambda df: df.black_wins + df.white_wins).rename(columns={'black_wins': 'wins'})[['wins', 'games']]

    trials = (black + white)
    trials = trials.groupby(trials.index).sum()

    return pd.concat({
        'best_games': trials.games,
        'best_elo': np.log(trials.wins + 1) - np.log(trials.games - trials.wins + 1)}, 1)


