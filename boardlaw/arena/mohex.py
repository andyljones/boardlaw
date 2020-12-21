from collections import deque
from boardlaw.arena import evaluator
import torch
import numpy as np
from .. import mohex, hex
from . import database, analysis
from rebar import arrdict, stats
from logging import getLogger
import activelo
import pandas as pd

log = getLogger(__name__)

BOARDSIZES = [3, 5, 7, 9, 11]
RUN_NAMES = [f'mohex-{s}' for s in BOARDSIZES]

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
    imp = activelo.improvement(soln)
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

    def __init__(self, worldfunc, max_history=256):
        self.worlds = worldfunc(8)
        self.mohex = mohex.MoHexAgent()
        self.history = deque(maxlen=max_history//self.worlds.n_envs)

    def trial(self, agent):
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
        stats.mean_std(μ, σ)

        imp = activelo.improvement(soln)
        imp = pd.DataFrame(imp, games.index, games.index)

        challenger = imp['agent'].idxmax()
        randomness = float(challenger.split('-')[1])
        self.mohex.random = randomness
        results = evaluator.evaluate(self.worlds, {'agent': agent, challenger: self.mohex})
        log.info(f'Agent played {challenger}, {int(results[0].wins[0] + results[1].wins[1])}-{int(results[0].wins[1] + results[1].wins[0])}')
        self.history.extend(results)
