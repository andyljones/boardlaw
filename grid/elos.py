import torch
import numpy as np
import pandas as pd

def symmetrize(trials, agents=None):
    df = (trials
            .assign(games=lambda df: df.black_wins + df.white_wins)
            .pivot('black_agent', 'white_agent', ['games', 'white_wins', 'black_wins'])).fillna(0)

    ws = (df.black_wins/df.games + df.white_wins/df.games.T)/2*(df.games + df.games.T)/2.
    gs = (df.games + df.games.T)/2.

    return ws, gs

def fast_elos(wins, games, prior=1):

    W = torch.as_tensor(wins.fillna(0).values) + prior
    N = torch.as_tensor(games.fillna(0).values) + 2*prior
    mask = torch.as_tensor(games.isnull().values)

    n = N.shape[0]
    r = torch.nn.Parameter(torch.zeros(n))

    def loss():
        d = r[:, None] - r[None, :]
        s = 1/(1 + torch.exp(-d))
        
        l = W*s.log() + (N - W)*(1 - s).log()
        return -l[~mask].mean() + .01*r.sum().pow(2)

    optim = torch.optim.LBFGS([r], line_search_fn='strong_wolfe', max_iter=100)

    def closure():
        l = loss()
        optim.zero_grad()
        l.backward()
        return l
        
    optim.step(closure)
    closure()

    return (r - r.max()).detach().cpu().numpy()

def elo_errors(snaps, trials):
    μ = snaps.μ

    ws, gs = symmetrize(trials, snaps.index)
    rates = (ws/gs).reindex(index=μ.index, columns=μ.index)

    diffs = pd.DataFrame(μ.values[:, None] - μ.values[None, :], μ.index, μ.index)
    expected = 1/(1 + np.exp(-diffs))

    err = (rates - expected).abs()
    return pd.concat([err.max(), err.T.max()], 1).max(1)
