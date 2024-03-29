import torch
import numpy as np
import pandas as pd

def symmetrize(trials):
    if len(trials) == 0:
        ws = pd.DataFrame(0., index=trials.index, columns=trials.index)
        gs = pd.DataFrame(0., index=trials.index, columns=trials.index)
        return ws, gs

    df = (trials
            .assign(games=lambda df: df.black_wins + df.white_wins)
            .groupby(['black_agent', 'white_agent'])[['games', 'white_wins', 'black_wins']].sum()
            .unstack())
    ids = list(set(df.columns.get_level_values(1)))
    df = df.reindex(index=ids).reindex(columns=ids, level=1).fillna(0)

    ws = (df.black_wins/df.games + df.white_wins.T/df.games.T)/2*(df.games + df.games.T)
    gs = (df.games + df.games.T)

    return ws.where(gs > 0, np.nan), gs

def solve(wins, games, prior=1):
    pd.testing.assert_index_equal(wins.index, games.index)
    pd.testing.assert_index_equal(wins.columns, games.columns)
    pd.testing.assert_index_equal(wins.index, wins.columns, check_names=False)

    W = torch.as_tensor(wins.fillna(0).values).double() + prior
    N = torch.as_tensor(games.fillna(0).values).double() + 2*prior
    mask = torch.as_tensor(games.gt(0).values)

    n = N.shape[0]
    r = torch.nn.Parameter(torch.zeros(n).double())

    def loss():
        d = r[:, None] - r[None, :]
        s = 1/(1 + torch.exp(-d))
        
        l = W*s.log() + (N - W)*(1 - s).log()
        loss = -l[mask].mean() + .01*r.mean().pow(2)
        if torch.isinf(loss):
            print('Infinite loss!')
        return loss

    optim = torch.optim.LBFGS([r], line_search_fn='strong_wolfe', max_iter=200)

    def closure():
        l = loss()
        optim.zero_grad()
        l.backward()
        return l
        
    optim.step(closure)
    optim.step(closure)
    closure()

    return pd.Series((r - r.max()).detach().cpu().numpy(), wins.index, name='elo')

def elo_errors(snaps, trials):
    μ = snaps.μ

    ws, gs = symmetrize(trials, snaps.index)
    rates = (ws/gs).reindex(index=μ.index, columns=μ.index)

    diffs = pd.DataFrame(μ.values[:, None] - μ.values[None, :], μ.index, μ.index)
    expected = 1/(1 + np.exp(-diffs))

    err = (rates - expected).abs()
    return pd.concat([err.max(), err.T.max()], 1).max(1)
