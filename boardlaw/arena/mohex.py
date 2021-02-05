from collections import deque
from boardlaw.arena import evaluator
import torch
import numpy as np
from .. import mohex, hex
from . import database, analysis
from rebar import arrdict
from pavlov import stats
from logging import getLogger
import activelo
import pandas as pd

log = getLogger(__name__)

BOARDSIZES = [3, 5, 7, 9, 11, 13]
RUN_NAMES = [f'mohex-{s}' for s in BOARDSIZES]

def activelo_refill(run_name, names, queue, count=1):
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

def run(boardsize):
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

            refill(run_name, names, queue)
            log.info(f'Starting on {queue[0]}')
            active[idx] = torch.tensor(queue[0])
            queue = queue[1:]

def all_elos():
    df = pd.concat({n: analysis.elos(n, target=-1) for n in RUN_NAMES}, 1)
    ax = df.xs('μ', 1, 1).plot()
    ax.invert_xaxis()

def total_games():
    return pd.Series({n: database.games(n).sum().sum() for n in RUN_NAMES})

def append(df, name):
    names = list(df.index) + [name]
    return df.reindex(index=names, columns=names).fillna(0)

class Trialer:

    def __init__(self, worldfunc, max_history=128):
        self.worlds = worldfunc(4)
        self.mohex = mohex.MoHexAgent()
        self.history = deque(maxlen=max_history//self.worlds.n_envs)

    def trial(self, agent, record=True):
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
        log.info(f'Agent elo is {μ:.2f}±{2*σ:.2f} based on {2*int(games.loc["agent"].sum())} games')
        if record:
            stats.mean_std('elo-mohex', μ, σ)

        imp = activelo.improvement(soln)
        imp = pd.DataFrame(imp, games.index, games.index)

        challenger = imp['agent'].idxmax()
        randomness = float(challenger.split('-')[1])
        self.mohex.random = randomness
        results = evaluator.evaluate(self.worlds, {'agent': agent, challenger: self.mohex})
        log.info(f'Agent played {challenger}, {int(results[0].wins[0] + results[1].wins[1])}-{int(results[0].wins[1] + results[1].wins[0])}')
        self.history.extend(results)

def judge(worldfunc, agent):
    trialer = Trialer(worldfunc)
    while True:
        trialer.trial(agent, record=False)

def plot(run):
    import matplotlib.pyplot as plt
    import copy

    games, wins = database.symmetric(run)
    games, wins = analysis.mask(games, wins, '.*')
    soln = activelo.solve(games.values, wins.values)

    rates = wins/games

    expected = 1/(1 + np.exp(-soln.μ[:, None] + soln.μ[None, :]))
    actual = rates.where(games > 0, np.nan).values

    fig = plt.figure()
    gs = plt.GridSpec(4, 3, fig, height_ratios=[20, 1, 20, 1])
    fig.set_size_inches(18, 12)

    # Top row
    cmap = copy.copy(plt.cm.RdBu)
    cmap.set_bad('lightgrey')
    kwargs = dict(cmap=cmap, vmin=0, vmax=1, aspect=1)

    ax = plt.subplot(gs[0, 0])
    ax.imshow(actual, **kwargs)
    ax.set_title('actual')

    ax = plt.subplot(gs[0, 1])
    im = ax.imshow(expected, **kwargs)
    ax.set_title('expected')

    ax = plt.subplot(gs[1, :2])
    plt.colorbar(im, cax=ax, orientation='horizontal')

    # Top right
    ax = plt.subplot(gs[0, 2])
    ax.errorbar(np.arange(len(soln.μ)), soln.μd[0, :], yerr=soln.σd[0, :], marker='.', capsize=2, linestyle='')
    ax.set_title('elos v. first')
    ax.grid()

    # Bottom left
    ax = plt.subplot(gs[2, 0])
    im = ax.imshow(actual - expected, vmin=-1, vmax=+1, aspect=1, cmap=cmap)
    ax.set_title('error')

    ax = plt.subplot(gs[3, 0])
    plt.colorbar(im, cax=ax, orientation='horizontal')
    # ax.annotate(f'resid var: {resid_var:.0%}, corr: {corr:.0%}', (.5, -1.2), ha='center', xycoords='axes fraction')

    # Bottom middle
    ax = plt.subplot(gs[2, 1])
    se = (expected*(1-expected)/games)**.5
    im = ax.imshow((actual - expected)/se, vmin=-3, vmax=+3, aspect=1, cmap='RdBu')
    ax.set_title('standard error')

    ax = plt.subplot(gs[3, 1])
    plt.colorbar(im, cax=ax, orientation='horizontal')
    # ax.annotate(f'resid var: {resid_var:.0%}, corr: {corr:.0%}', (.5, -1.2), ha='center', xycoords='axes fraction')