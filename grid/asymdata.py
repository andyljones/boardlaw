import cvxpy as cp
import numpy as np
import torch
from contextlib import contextmanager
import pandas as pd
from pkg_resources import resource_filename
from pavlov import json, runs
from pathlib import Path
import json as json_

PREFIX = 'arena'

KEYS = ['black_name', 'white_name']

def _to_dict(l):
    # This indirection is because we can't store multi-part keys in a JSON. Ugh.
    return {tuple(r[n] for n in KEYS): {k: v for k, v in r.items() if k not in KEYS} for r in l}

def _to_list(d):
    return [{**dict(zip(KEYS, k)), **v} for k, v in d.items()]

def boardsize_path(boardsize):
    return Path(f'output/experiments/eval/asym/{boardsize}.json')

def assure(boardsize):
    path = boardsize_path(boardsize)
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text('[]')

@contextmanager
def update(boardsize):
    p = boardsize_path(boardsize)
    contents = json_.loads(p.read_text())
    yield contents
    p.write_text(json_.dumps(contents))

def save(results):
    if not results:
        return
    
    boardsize = results[0].boardsize
    
    assure(boardsize)
    with update(boardsize) as l:
        d = _to_dict(l)
        for result in results:
            k = tuple(result.names)
            if k not in d:
                d[k] = {'black_wins': 0, 'white_wins': 0, 'moves': 0, 'times': 0.}
            v = d[k]
            v['black_wins'] += result.wins[0]
            v['white_wins'] += result.wins[1]
            v['moves'] += result.moves
            v['times'] += result.times

        l[:] = _to_list(d)

def pandas(boardsize):
    path = boardsize_path(boardsize)
    if path.exists():
        contents = json_.loads(path.read_text())
    else:
        contents = []

    if contents:
        return pd.DataFrame(contents).set_index(KEYS)
    else:
        return pd.DataFrame(columns=['black_name', 'white_name', 'black_wins', 'white_wins', 'moves']).set_index(KEYS)

def symmetrize(raw, agents=None):
    games = (raw.black_wins + raw.white_wins).unstack().reindex(index=agents, columns=agents).fillna(0)

    black_wins = raw.black_wins.unstack().reindex_like(games)
    white_wins = raw.white_wins.unstack().reindex_like(games).T

    ws = (black_wins/games + white_wins/games.T)/2*(games + games.T)/2.
    gs = (games + games.T)/2.

    return ws, gs

def fast_elos(ws, gs, prior=1):

    W = torch.as_tensor(ws.fillna(0).values) + prior
    N = torch.as_tensor(gs.fillna(0).values) + 2*prior

    n = N.shape[0]
    r = torch.nn.Parameter(torch.zeros(n))

    def loss():
        d = r[:, None] - r[None, :]
        s = 1/(1 + torch.exp(-d))
        
        l = W*s.log() + (N - W)*(1 - s).log()
        return -l.mean() + r.sum().pow(2)

    optim = torch.optim.LBFGS([r], line_search_fn='strong_wolfe')

    def closure():
        l = loss()
        optim.zero_grad()
        l.backward()
        return l
        
    optim.step(closure)
    closure()

    return (r - r.max()).detach().cpu().numpy()

def elo_errors(snaps, raw):
    μ = snaps.μ

    ws, gs = symmetrize(raw, snaps.index)
    rates = (ws/gs).reindex(index=μ.index, columns=μ.index)

    diffs = pd.DataFrame(μ.values[:, None] - μ.values[None, :], μ.index, μ.index)
    expected = 1/(1 + np.exp(-diffs))

    err = (rates - expected).abs()
    return pd.concat([err.max(), err.T.max()], 1).max(1)

def pandas_elos(boardsize, **kwargs):
    from . import data

    snaps = data.snapshot_solns(boardsize, solve=False)
    raw = pandas(boardsize)
    ws, gs = symmetrize(raw, snaps.index)
    snaps['μ'] = fast_elos(ws, gs, **kwargs)
    snaps['err'] = elo_errors(snaps, raw)

    return snaps

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

def nash_solns(snaps, ws, gs):
    rate = (ws/gs).fillna(.5)

    strats = {}
    payoffs = {}
    for i, i_group in snaps.reindex(rate.index).groupby('idx'):
        for j, j_group in snaps.reindex(rate.index).groupby('idx'):
            if i != j:
                A = rate.reindex(index=i_group.index, columns=j_group.index)
                x, y = equilibrium(A)
                strats[i, j] = x, y 
                payoffs[i, j] = x @ A @ y
    payoffs = pd.Series(payoffs).unstack()
    return strats, payoffs

def nash_elos(payoffs):
    ws = 1000*payoffs.fillna(.5)
    gs = 0*ws + 1000

    return pd.Series(fast_elos(ws, gs), payoffs.index)

def plot_nash_elos(joint):
    import plotnine as pn
    from . import plot
    import statsmodels.formula.api as smf

    cutoffs = {
        3: (5e10, 2e11),
        4: (1e11, 2e12),
        5: (2e11, 8e12),
        6: (7e11, 2e14),
        7: (1e12, 3e15),
        8: (3e12, 1e16),
        9: (6e12, 1e17)}
    cutoffs = pd.DataFrame.from_dict(cutoffs, orient='index', columns=('lower', 'upper')).rename_axis(index='boardsize')

    df =(
        joint
        .join(cutoffs, on='boardsize')
        .query('lower <= flops <= upper'))

    model = smf.ols('elo ~ boardsize + np.log10(flops) + 1', df).fit()
    df['elohat'] = model.predict(df)

    ps = model.params.mul(400/np.log(10)).apply(lambda x: f'{float(f"{x:.2g}"):g}')
    s = f'$\mathrm{{elo}} = {ps.boardsize} \cdot \mathrm{{boardsize}} + {ps["np.log10(flops)"]} \cdot \ln_{{10}}(\mathrm{{flops}}) + C$'

    return (pn.ggplot(df)
        + pn.geom_line(pn.aes(x='flops', y='400/np.log(10)*elo', group='boardsize', color='factor(boardsize)'), size=2)
        + pn.geom_line(pn.aes(x='flops', y='400/np.log(10)*elohat', color='factor(boardsize)'), size=1, linetype='dashed')
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_discrete(name='boardsize')
        + pn.labs(title='elos of nash equilibiria')
        + pn.annotate('text', 3e14, -2000, label=s, size=20)
        + pn.coord_cartesian(None, (None, 0))
        + plot.mpl_theme()
        + plot.poster_sizes())


def run_nash_elos():
    snaps = pd.concat([pandas_elos(b) for b in range(3, 10)])
    snaps = snaps[snaps.nodes.isnull()]

    elos = {}
    for b in snaps.boardsize.unique():
        print(b)
        raw = pandas(b)
        ws, gs = symmetrize(raw)
        strats, payoffs = nash_solns(snaps, ws, gs)
        elos[b] = nash_elos(payoffs)
    elos = pd.concat(elos).rename('elo').reset_index().rename(columns={'level_0': 'boardsize', 'level_1': 'idx'})
    flops = snaps.groupby(['boardsize', 'idx']).flops.mean().reset_index()
    joint = pd.merge(elos, flops)

    return plot_nash_elos(joint)

def plot_adv_v_flops(snaps):
    import plotnine as pn
    from . import plot
    
    diffs = {}
    for b in range(3, 10):
        print(b)
        raw = pandas(b)
        ws, gs = symmetrize(raw)

        strats, payoffs = nash_solns(snaps, ws, gs)

        log_flops = snaps[snaps.boardsize == b].groupby('idx').flops.apply(lambda g: np.log10(g).mean())
        index = (log_flops.values[1:] + log_flops.values[:-1])/2
        diffs[b] = pd.Series((np.diag(payoffs, -1) - np.diag(payoffs, 1))/2, 10**index)
        
    df = pd.concat(diffs).reset_index()
    df.columns = ['boardsize', 'flops', 'adv']

    return (pn.ggplot(df)
        + pn.geom_line(pn.aes(x='flops', y='adv', color='factor(boardsize)'), size=2)
        + pn.scale_x_continuous(trans='log10')
        + plot.mpl_theme()
        + plot.poster_sizes())