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
from . import plot, data
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

def plot_flops_frontier(ags):
    df = data.modelled_elos(ags)

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

def plot_resid_var_trends(ags):
    resid_var = data.residual_vars(ags)
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