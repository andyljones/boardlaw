import pandas as pd
import numpy as np
from . import plot, data, overleaf
import plotnine as pn
import matplotlib.patheffects as path_effects
from boardlaw import arena, analysis
from functools import wraps
import torch

RUNS = {
    3: ('2021-02-17 21-01-19 arctic-ease', 20),
    9: ('2021-02-20 23-35-25 simple-market', 20)}

# All Elos internally go as e^d; Elos in public are in base 10^(d/400)
ELO = 400/np.log(10)

def upload(f, *args, **kwargs):
    [_, name] = f.__name__.split('plot_')

    y = f(*args, **kwargs)
    overleaf.plot(y, name + '.png')

def plot_hex(n_envs=1, boardsize=9, seed=8):
    torch.manual_seed(seed)

    run, idx = RUNS[boardsize]

    world = arena.common.worlds(run, n_envs)
    agent = arena.common.agent(run, idx)
    trace = analysis.rollout(world, [agent, agent], n_trajs=1)

    penult = trace.worlds[-2]
    actions = trace.actions[-1]
    ult, _ = penult.step(actions, reset=False)

    return ult.display()

def plot_flops_curves(ags):
    df = ags.query('test_nodes == 64').copy()
    df['g'] = ags.run
    labels = df.sort_values('train_flops').groupby('boardsize').first().reset_index()

    return (pn.ggplot(df, pn.aes(x='train_flops', color='factor(boardsize)', group='g'))
        + pn.geom_line(pn.aes(y='ELO*elo'), size=.25, show_legend=False)
        + pn.geom_point(pn.aes(y='ELO*elo'), size=1/16, show_legend=False, shape='.')
        + pn.geom_text(pn.aes(y='ELO*elo', label='boardsize'), data=labels, show_legend=False, size=6, nudge_x=-.25, nudge_y=-15)
        + pn.labs(
            x='FLOPS', 
            y='Elo v. perfect play')
        + pn.scale_color_discrete(l=.4)
        + pn.scale_x_continuous(trans='log10')
        + pn.coord_cartesian(None, (None, 0))
        + plot.IEEE())

def plot_frontiers(ags):
    df, model = data.modelled_elos(ags)
    labels = df.sort_values('train_flops').groupby('boardsize').first().reset_index()

    return (pn.ggplot(df, pn.aes(x='train_flops', color='factor(boardsize)', group='boardsize'))
                + pn.geom_line(pn.aes(y='ELO*elo'), size=.5, show_legend=False)
                + pn.geom_line(pn.aes(y='ELO*elohat'), size=.25, linetype='dashed', show_legend=False)
                + pn.geom_text(pn.aes(y='ELO*elohat', label='boardsize'), data=labels, show_legend=False, size=6, nudge_x=-.25, nudge_y=-15)
                + pn.labs(
                    x='FLOPS', 
                    y='Elo v. perfect play')
                + pn.scale_color_discrete(l=.4)
                + pn.scale_x_continuous(trans='log10')
                + pn.coord_cartesian(None, (None, 0))
                + plot.IEEE())

def plot_resid_var(ags):
    resid_var = data.residual_vars(ags)
    resid_var['diff'] = resid_var.predicted - resid_var.seen
    labels = resid_var.sort_values('seen').groupby('predicted').last().reset_index()
    return (pn.ggplot(resid_var, pn.aes(x='seen', y='rv', color='factor(predicted)', group='predicted'))
        + pn.geom_line(size=.25, show_legend=False)
        + pn.geom_text(pn.aes(label='predicted'), labels, nudge_x=+.15, size=6, show_legend=False)
        + pn.geom_point(size=.25, show_legend=False)
        + pn.scale_y_continuous(trans='log10')
        + pn.scale_color_discrete(l=.4)
        + pn.labs(
            x='max board size observed',
            y='residual variance')
        + plot.IEEE())

def plot_runtimes(ags):
    threshold = -50/(400/np.log(10))
    best = (ags
        .query('test_nodes == 64')
        .loc[lambda df: df.elo > threshold]
        .sort_values('train_time')
        .groupby('boardsize').first()
        .reset_index())
    return (pn.ggplot(best, pn.aes(x='boardsize', y='train_time'))
        + pn.geom_point(size=.5)
        + pn.geom_line(size=.5)
        + pn.scale_y_continuous(trans='log10')
        + pn.labs(x='Board size', y='Training time (s)')
        + plot.IEEE())

def plot_train_test(ags):
    df = ags.query('boardsize == 9').copy()
    df['test_flops'] = df.test_nodes*(df.train_flops/df.samples)
    df['train_flops_group'] = df.train_flops.pipe(np.log10).round(1).pipe(lambda s: 10**s)

    frontiers = {}
    for e in np.linspace(-1500, 0, 7):
        frontiers[e] = df[ELO*df.elo > e].groupby('train_flops_group').test_flops.min().expanding().min()
    frontiers = pd.concat(frontiers).unstack().T

    distinct = frontiers.pipe(np.log10).round(1).pipe(lambda df: 10**df)
    distinct = distinct.where(distinct.iloc[-1].eq(distinct).cumsum().le(1))
    distinct = distinct.stack().reset_index().sort_values('train_flops_group')
    distinct.columns = ['train_flops', 'elo', 'test_flops']

    labs = distinct.sort_values('train_flops').groupby('elo').first().reset_index()
    return (pn.ggplot(distinct, pn.aes(x='train_flops', y='test_flops', color='elo', group='elo'))
        + pn.geom_line(size=.5, show_legend=False)
        + pn.geom_point(data=distinct, size=.5, show_legend=False)
        + pn.geom_text(pn.aes(label='elo.astype(int)'), labs, show_legend=False, size=6, nudge_y=+.2)
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_y_continuous(trans='log10')
        + pn.labs(
            x='train-time FLOPS',
            y='test-time FLOPS')
        + plot.IEEE())


def boardsize_hyperparams_table(ags):
    return (ags
        .groupby('boardsize')
        [['width', 'depth', 'samples', 'train_flops']]
        .max()
        .assign(train_flops=lambda df: df.train_flops.apply(lambda s: f'{s:.1G}'))
        .assign(samples=lambda df: df.samples.apply(lambda s: f'{s:.1G}'))
        .rename(columns={'boardsize': 'board size', 'width': 'max neurons', 'depth': 'max layers', 'samples': 'max samples', 'train_flops': 'max flops'})
        .reset_index()
        .to_latex(index=True, label='boardsize', caption='Board size-dependent hyperparameters'))

def parameters_table(ags):
    df, model = data.modelled_elos(ags)
    params = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
    raw = ELO*pd.Series({
            ('$m_\text{boardsize}$', 'plateau'): params['plateau'][0],
            ('$c$', 'plateau'): params['plateau'][1],
            ('$m_\text{flops}$', 'incline'): params['incline'][0],
            ('$m_\text{boardsize}$', 'incline'): params['incline'][1],
            ('$c$', 'incline'): params['incline'][2]})

    return (raw
            .apply(plot.sig_figs, n=2)
            .unstack(0)
            .fillna('')
            .iloc[::-1, ::-1]
            .to_latex(index=True, label='parameters', caption='Fitted frontier parameters', escape=False))

if __name__ == '__main__':
    ags = data.load()
    #TODO: Push this back into the database
    ags = data.with_times(ags)
    upload(plot_hex)
    upload(plot_flops_curves, ags)
    upload(plot_frontiers, ags)
    upload(plot_resid_var, ags)
    upload(plot_runtimes, ags)
    upload(plot_train_test, ags)

    overleaf.table(boardsize_hyperparams_table(ags), 'boardsize_hyperparams')
    overleaf.table(parameters_table(ags), 'parameters')