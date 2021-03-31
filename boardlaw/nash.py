import cvxpy as cp
import pandas as pd
import numpy as np
from . import elos, sql, storage
from tqdm.auto import tqdm

def equilibrium(A):
    # https://www.cs.cmu.edu/~ggordon/780-spring09/slides/Game%20theory%20lecture2-algs%20for%20normal%20form.pdf
    x = cp.Variable(A.shape[0])
    z = cp.Variable()

    bounds = [
        z - A.values.T @ x <= 0, 
        cp.sum(x) == 1, 
        x >= 0]

    cp.Problem(cp.Maximize(z), bounds).solve()

    x = pd.Series(x.value, A.index)
    y = pd.Series(bounds[0].dual_value, A.columns)
    return x, y

def nash_solns(ags, ws, gs):
    [boardsize] = ags.boardsize.unique()
    schedule = storage.savepoints(boardsize)
    assignment = abs(np.log10(ags.train_flops.reindex(ws.index).values)[:, None] - np.log10(schedule)[None, :]).argmin(1)

    rate = (ws/gs).fillna(.5)

    strats = {}
    payoffs = {}
    for i in range(len(schedule)):
        for j in range(len(schedule)):
            iflops, jflops = schedule[i], schedule[j]
            A = rate.loc[assignment == i].loc[:, assignment == j]
            if (i != j) and (A.size > 0):
                x, y = equilibrium(A)
                strats[iflops, jflops] = x, y 
                payoffs[iflops, jflops] = x @ A @ y
    payoffs = pd.Series(payoffs).unstack()
    return strats, payoffs

def nash_elos(payoffs):
    ws = 1000*payoffs.fillna(.5)
    gs = 0*ws + 1000

    return pd.Series(elos.solve(ws, gs), payoffs.index)

def incremental_solns(ags, ws, gs):
    [boardsize] = ags.boardsize.unique()
    schedule = np.log10(storage.savepoints(boardsize))
    cutoffs = (schedule[1:] + schedule[:-1])/2
    cutoffs = np.concatenate([cutoffs, [np.inf]])

    flops = ags.train_flops.reindex(ws.index).pipe(np.log10)
    assignments = (flops.values[:, None] < cutoffs[None, :]).argmax(1)

    rate = (ws/gs).fillna(.5)

    payoffs = {}
    for i in range(len(schedule)):
        A = rate.loc[assignments <= i]
        if (A.size > 0):
            x, y = equilibrium(A)
            payoffs[schedule[i]] = x @ A @ y
    payoffs = pd.Series(payoffs)

    return payoffs

def nash_payoffs(boardsize=None):
    if boardsize is None:
        return pd.concat({b: nash_payoffs(b) for b in tqdm(range(3, 10))}, names=('boardsize',)).reset_index(level=0).reset_index(drop=True)
    ags = (sql.agent_query()
            .query('test_nodes == 64')
            .loc[lambda df: df.description == f'bee/{boardsize}'])
    trials = (sql
                .trial_query(boardsize)
                .loc[lambda df: df.black_agent.isin(ags.index) & df.white_agent.isin(ags.index)])

    ws, gs = elos.symmetrize(trials)
    strats, payoffs = nash_solns(ags, ws, gs)

    payoffs = payoffs.stack().reset_index()
    payoffs.columns = ['strategy', 'challenger', 'winrate']

    return payoffs

def nash_grads(payoffs):
    grads = {}
    for boardsize, g in payoffs.groupby('boardsize'):
        df = g.pivot('strategy', 'challenger', 'winrate')
        mid_up = np.log10(df.index)[1:]
        mid_down = np.log10(df.index)[:-1]
        dy = np.diag(df, -1) - np.diag(df, +1)
        dx = mid_up - mid_down
        grads[boardsize] = pd.Series(dy/dx, 10**((mid_up + mid_down)/2))
    grads = pd.concat(grads, names=('boardsize',)).rename('grad').reset_index()
    return grads