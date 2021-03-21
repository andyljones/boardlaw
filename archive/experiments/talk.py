import scipy as sp
import torch
from torch import nn
import aljpy
import hashlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import aljpy.plot
from pavlov import runs
from boardlaw import analysis, arena
from grid import sql, plot, elos
from tqdm.auto import tqdm
import plotnine as pn
from rebar import arrdict

RUNS = {
    3: ('2021-02-17 21-01-19 arctic-ease', 20),
    9: ('2021-02-20 23-35-25 simple-market', 20)}

def record_games(n_envs=1, boardsize=9):
    run, idx = RUNS[boardsize]

    world = arena.common.worlds(run, n_envs)
    agent = arena.common.agent(run, idx)
    trace = analysis.rollout(world, [agent, agent], n_trajs=1)

    # Bit of a mess this
    if n_envs == 1:
        penult = trace.worlds[-2]
        actions = trace.actions[-1]

        ult, _ = penult.step(actions, reset=False)
        augmented = arrdict.cat([trace.worlds[[-1]], trace.worlds[:-1], ult[None], ult[None]], 0)
    else:
        augmented = trace.worlds

    return analysis.record_worlds(augmented).notebook()

def plot_elo_winrates():
    diffs = np.linspace(-1000, +1000)
    rates = 1/(1 + 10**(-(diffs/400)))
    with plt.style.context('seaborn-poster'):
        fig, ax = plt.subplots()
        ax.plot(diffs, rates)
        ax.set_ylim(0, 1)
        ax.set_xlim(-1000, +1000)
        ax.grid(True)
        ax.axhline(0.5, color='k', alpha=.5)
        ax.axvline(0, color='k', alpha=.5)
        ax.set_ylabel('win rate')
        ax.set_xlabel('difference in Elos')
        ax.set_yticks([0, .25, .5, .75, 1.])
        aljpy.plot.percent_axis(ax, axis='y')
        ax.set_title('win rate is a sigmoid in rating difference')

@aljpy.autocache()
def _trial_elos(boardsize, counter):
    trials = sql.trial_query(boardsize, 'bee/%')
    ws, gs = elos.symmetrize(trials)
    return elos.solve(ws, gs)

def trial_elos(boardsize):
    counter = sql.file_change_counter()
    return _trial_elos(boardsize, counter)

def load():
    ags = sql.agent_query()

    es = []
    for b in tqdm(ags.boardsize.unique()):
        es.append(trial_elos(b))
    es = pd.concat(es)

    return ags.join(es, how='inner')

def plot_training_curves(ags):
    df = ags[ags.test_nodes == 64].copy()
    df['g'] = df.run + df.test_nodes.astype(str)

    return (pn.ggplot(df, pn.aes(x='train_flops', y='400/np.log(10)*elo', group='g', color='factor(boardsize)'))
            + pn.geom_line()
            + pn.geom_point(size=.5)
            + pn.scale_x_continuous(trans='log10')
            + pn.scale_color_discrete(name='Boardsize')
            + pn.labs(
                x='Training FLOPS', 
                y='Elo v. perfect play',
                title='All agents\' training curves')
            + plot.mpl_theme()
            + plot.poster_sizes())

def interp_frontier(g, x='train_flops', y='elo', group='run'):
    xl, xr = g[x].pipe(np.log10).min(), g[x].pipe(np.log10).max()
    xs = np.linspace(xl, xr, 101)
    ys = {}
    for run, gg in g.sort_values(x).groupby(group):
        xp = gg[x].pipe(np.log10).values
        yp = gg[y].values
        ys[run] = np.interp(xs, xp, yp, np.nan, np.nan)
    ys = pd.DataFrame(ys, index=10**xs)

    return ys.max(1).rename_axis(index=x).rename(y)

class Changepoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.lower = nn.Parameter(torch.as_tensor([-1.5, 3.]))
        self.linear = nn.Parameter(torch.as_tensor([2., -2, -16]))
        
    def forward(self, X):
        X = torch.cat([X, torch.ones_like(X[:, :1])], -1)
        lower = X[:, 1:] @ self.lower
        linear = X @ self.linear
        return torch.maximum(linear, lower).clamp(None, 0)

class Sigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.vscale = nn.Parameter(torch.as_tensor([1.3, -2.]))
        self.hscale = nn.Parameter(torch.as_tensor([1/16., 0.]))
        self.center = nn.Parameter(torch.as_tensor([.66, 9.]))
        
    def forward(self, X):
        X = torch.cat([X, torch.ones_like(X[:, :1])], -1)
        vscale = X[:, 1:] @ self.vscale
        hscale = X[:, 1:] @ self.hscale
        center = X[:, 1:] @ self.center
        return vscale*(torch.sigmoid((X[:, 0] - center)/hscale) - 1)

def model_inputs(df):
    return torch.stack([
        torch.as_tensor(df.train_flops.values).log10().float(),
        torch.as_tensor(df.boardsize.values).float(),], -1)

def fit_model(df):
    X = model_inputs(df)
    y = torch.as_tensor(df.elo.values)

    model = Sigmoid()
    optim = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe', max_iter=100)

    def closure():
        yhat = model(X)
        loss = (y - yhat).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        return loss
        
    optim.step(closure)

    return model

def apply_model(model, df):
    X = model_inputs(df)
    return pd.Series(model(X).detach().cpu().numpy(), df.index)
        
def plot_flops_frontier(ags, b=np.inf):
    df = (ags.query('test_nodes == 64')
            .groupby('boardsize')
            .apply(interp_frontier, 'train_flops')
            .reset_index()) 
    
    model = fit_model(df[df.boardsize <= b])
    df['elohat'] = apply_model(model, df)

    return (pn.ggplot(df, pn.aes(x='train_flops', color='factor(boardsize)', group='boardsize'))
                + pn.geom_line(pn.aes(y='400/np.log(10)*elo'), size=2)
                + pn.geom_line(pn.aes(y='400/np.log(10)*elohat'), size=1, linetype='dashed')
                + pn.labs(
                    x='Training FLOPS', 
                    y='Elo v. perfect play',
                    title='Performance is a sigmoid of compute, linearly scaled by board size')
                + pn.scale_x_continuous(trans='log10')
                + pn.scale_color_discrete(name='Boardsize')
                + pn.coord_cartesian(None, (None, 0))
                + plot.mpl_theme()
                + plot.poster_sizes())
    
def perfect_play(model, target=-100):
    perfect = {}
    for b in range(3, 10):
        f = lambda x: 400/np.log(10)*model(torch.as_tensor([[x, b]])).detach().numpy().squeeze() - target
        perfect[b] = sp.optimize.bisect(f, 1, 18)
    return pd.Series(perfect, name='perfect')

def plot_resid_var_trends(ags):
    df = (ags.query('test_nodes == 64')
            .groupby('boardsize')
            .apply(interp_frontier, 'train_flops')
            .reset_index()) 

    yhats = {}
    for b in range(4, 10):
        model = fit_model(df[df.boardsize <= b])
        yhats[b] = apply_model(model, df[df.boardsize >= b])
    yhats = pd.concat(yhats, 1)

    num = yhats.sub(df.elo, 0).pow(2).groupby(df.boardsize).mean()
    denom = df.elo.pow(2).groupby(df.boardsize).mean()
    resid_var = (num/denom).stack().reset_index()
    resid_var.columns = ['predicted', 'seen', 'rv']

    perfect = perfect_play(model)
    resid_var = (resid_var
        .join(perfect.rename('predicted_perfect'), on='predicted')
        .join(perfect.rename('seen_perfect'), on='seen')
        .assign(ratio=lambda df: 10**(df.seen_perfect - df.predicted_perfect)))

    return (pn.ggplot(resid_var, pn.aes(x='ratio', y='rv', color='factor(predicted)', group='predicted'))
        + pn.geom_line(size=2)
        + pn.geom_text(pn.aes(label='seen'), nudge_y=-.1, size=14)
        + pn.geom_point(size=4)
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_y_continuous(trans='log10')
        + pn.scale_color_discrete(name='Predicted frontier')
        + pn.labs(
            x='(cost of observed frontier)/(cost of predicted frontier)',
            y='residual variance in performance',
            title='Frontiers of small problems are good, cheap proxies for frontiers of expensive problems')
        + plot.mpl_theme()
        + plot.poster_sizes())

def plot_params(ags):
    df = ags.query('boardsize == 9 & test_nodes == 64').copy()
    df['params'] = df.train_flops/df.samples
    return (pn.ggplot(df, pn.aes(x='train_flops', y='400/np.log(10)*elo', color='params', group='run'))
        + pn.geom_line()
        + pn.geom_point()
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_continuous(trans='log10', name='Params')
        + pn.labs(
            title='Smaller networks are more compute efficient for lower performances, but plateau earlier',
            y='Elo v. perfect play',
            x='Train FLOPS')
        + plot.mpl_theme()
        + plot.poster_sizes()
        + plot.no_colorbar_ticks())

def plot_sample_efficiency(ags):
    df = ags.query('boardsize == 9 & test_nodes == 64').copy()
    df['params'] = df.train_flops/df.samples
    return (pn.ggplot(df, pn.aes(x='samples', y='400/np.log(10)*elo', color='params', group='run'))
        + pn.geom_line()
        + pn.geom_point()
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_continuous(trans='log10', name='Params')
        + pn.labs(
            title='Bigger networks might not be comute efficient, but they are sample efficient',
            y='Elo v. perfect play',
            x='Train FLOPS')
        + plot.mpl_theme()
        + plot.poster_sizes()
        + plot.no_colorbar_ticks())