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

def record_games():
    rs = runs.pandas().dropna()
    run = rs[rs.params.apply(lambda r: r['boardsize'] == 9)].index[-1]

    world = arena.common.worlds(run, 49)
    agent = arena.common.agent(run)
    analysis.record(world, [agent, agent], n_trajs=1).notebook()

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

def fit_model(df):
    X = torch.stack([
        torch.as_tensor(df.train_flops).log10().float(),
        torch.as_tensor(df.boardsize).float(),], -1)

    y = torch.as_tensor(df.elo)

    model = Sigmoid()
    optim = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe', max_iter=100)

    def closure():
        yhat = model(X)
        loss = (y - yhat).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        print(loss)
        return loss
        
    optim.step(closure)

    elohat = pd.Series(model(X).detach().cpu().numpy(), df.index)

    return model, elohat
        
def plot_flops_frontier(ags):
    df = (ags.query('test_nodes == 64')
            .groupby('boardsize')
            .apply(interp_frontier, 'train_flops')
            .reset_index()) 
    
    model, df['elohat'] = fit_model(df)

    return (pn.ggplot(df, pn.aes(x='train_flops', color='factor(boardsize)', group='boardsize'))
                + pn.geom_line(pn.aes(y='elo'), size=2)
                + pn.geom_line(pn.aes(y='elohat'), size=1, linetype='dashed')
                + pn.labs(
                    x='training flops', 
                    y='elo v. perfect play',
                    title='performance frontier in terms of compute')
                + pn.scale_x_continuous(trans='log10')
                + pn.scale_color_discrete(name='boardsize')
                + pn.coord_cartesian(None, (None, 0))
                + plot.mpl_theme()
                + plot.poster_sizes())