import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
from . import plot, data, overleaf
from .data import ELO
import plotnine as pn
import matplotlib.patheffects as path_effects
from boardlaw import analysis, elos
from boardlaw.arena import best
from functools import wraps
import torch
from mizani.formatters import percent_format

RUNS = {
    3: ('2021-02-17 21-01-19 arctic-ease', 20),
    9: ('2021-02-20 23-35-25 simple-market', 20)}

def upload(f, *args, **kwargs):
    [_, name] = f.__name__.split('plot_')

    y = f(*args, **kwargs)
    overleaf.plot(y, name + '.pdf')

def plot_hex(n_envs=1, boardsize=9, seed=8):
    torch.manual_seed(seed)
    from boardlaw import arena

    run, idx = RUNS[boardsize]

    world = arena.common.worlds(run, n_envs)
    agent = arena.common.agent(run, idx)
    trace = analysis.rollout(world, [agent, agent], n_trajs=1)

    penult = trace.worlds[-2]
    actions = trace.actions[-1]
    ult, _ = penult.step(actions, reset=False)

    return ult.display()

def plot_elos():
    diffs = np.linspace(-1000, +1000)
    rates = 1/(1 + 10**(-diffs/400))
    df = pd.DataFrame({'elo': diffs, 'winrate': rates})

    return (pn.ggplot(df)
        + pn.geom_line(pn.aes(x='elo', y='winrate'))
        + pn.geom_vline(xintercept=0, alpha=.1)
        + pn.geom_hline(yintercept=.5, alpha=.1)
        + pn.labs(
            x='Own Elo relative to opponent\'s Elo',
            y='Win rate v. opponent')
        + pn.scale_y_continuous(labels=percent_format())
        + pn.coord_cartesian(expand=False)
        + plot.IEEE())

def plot_flops_curves(ags):
    df = ags.query('test_nodes == 64').copy()
    labels = df.sort_values('train_flops').groupby('boardsize').first().reset_index()
    modelled, _ = data.modelled_elos(ags)

    return (pn.ggplot(df, pn.aes(x='train_flops', color='factor(boardsize)'))
            + pn.geom_line(pn.aes(y='ELO*elo', group='run'), size=.25, show_legend=False, alpha=.15)
            + pn.geom_line(pn.aes(y='ELO*elo'), modelled, size=.5, show_legend=False)
            + pn.geom_text(pn.aes(y='ELO*elo', label='boardsize'), data=labels, show_legend=False, size=6, nudge_x=-.25, nudge_y=-15)
            + pn.labs(
                x='Training compute (FLOPS-seconds)', 
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
                    x='Training compute (FLOPS-seconds)', 
                    y='Elo v. perfect play')
                + pn.scale_color_discrete(l=.4)
                + pn.scale_x_continuous(trans='log10')
                + pn.coord_cartesian(None, (None, 0))
                + plot.IEEE())

def plot_direct_frontiers(ags):
    ags = ags.copy()
    ags['elo'] = pd.concat([best.best_rates(best.TOPS[b]) for b in best.TOPS]).best_elo
    return plot_frontiers(ags)

def plot_resid_var(ags):
    resid_var = data.residual_vars(ags)
    resid_var['diff'] = resid_var.predicted - resid_var.seen
    labels = resid_var.sort_values('seen').groupby('predicted').last().reset_index()
    return (pn.ggplot(resid_var, pn.aes(x='seen', y='rv', color='factor(predicted)', group='predicted'))
        + pn.geom_line(size=.5, show_legend=False)
        + pn.geom_text(pn.aes(label='predicted'), labels, nudge_x=+.15, size=6, show_legend=False)
        + pn.geom_point(size=.5, show_legend=False)
        + pn.scale_y_continuous(trans='log10')
        + pn.scale_color_discrete(l=.4, limits=list(range(3, 10)))
        + pn.labs(
            x='Max board size observed',
            y='Residual Variance')
        + plot.IEEE())

def plot_runtimes(ags):
    threshold = -50/(400/np.log(10))
    best = (ags
        .query('test_nodes == 64')
        .loc[lambda df: df.elo > threshold]
        .sort_values('train_time')
        .groupby('boardsize').first()
        .reset_index())
    return (pn.ggplot(best, pn.aes(x='boardsize', y='train_time', color='factor(boardsize)'))
        + pn.geom_point(size=2, show_legend=False)
        # + pn.geom_line(size=.5, show_legend=False)
        + pn.scale_y_continuous(trans='log10')
        + pn.scale_color_discrete(l=.4)
        + pn.labs(x='Board size', y='Training time (seconds)')
        + plot.IEEE())

def plot_test(ags):
    df = ags.query('boardsize == 9').groupby('run').apply(lambda df: df[df.idx == df.idx.max()]).copy()
    df['test_flops'] = df.test_nodes*(df.train_flops/df.samples)

    subset = df.query('test_nodes == 64').sort_values('test_flops')
    selection = [subset.loc[ELO*subset.elo > e].iloc[0].run for e in np.linspace(-2000, -500, 4)]

    df = df[df.run.isin(selection)].copy()

    df['params'] = df.width**2 * df.depth
    df['arch'] = df.apply(lambda r: '{depth}×{width}'.format(**r), axis=1)
    labels = df.sort_values('test_flops').reset_index(drop=True).groupby('run').first().reset_index()
    return (pn.ggplot(df, pn.aes(x='test_flops', y='ELO*elo', color='params', group='run'))
        + pn.geom_point(size=.25, show_legend=False)
        + pn.geom_line(size=.5, show_legend=False)
        + pn.geom_text(pn.aes(label='test_nodes'), nudge_y=-50, show_legend=False, size=4, va='top')    + pn.geom_text(pn.aes(label='test_nodes'), nudge_y=-50, show_legend=False, size=4, va='top')
        + pn.geom_text(pn.aes(label='arch'), data=labels, show_legend=False, size=6, nudge_x=-.1, ha='right')
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_cmap('plasma', trans='log10', limits=(df.params.min(), 10*df.params.max()))
        + pn.coord_cartesian((3.5, None))
        + pn.labs(
            x='Test-time compute (FLOPS-seconds)',
            y='Elo v. perfect play')
        + plot.IEEE())

def plot_train_test(ags):
    frontiers = data.train_test(ags)
    frontiers, model = data.train_test_model(frontiers)
    labs = frontiers.sort_values('train_flops').groupby('elo').first().reset_index()
    return (pn.ggplot(frontiers, pn.aes(x='train_flops', y='test_flops', color='elo', group='elo'))
        + pn.geom_line(size=.5, show_legend=False)
        + pn.geom_line(pn.aes(y='test_flops_hat'), size=.25, show_legend=False, linetype='dashed')
        # + pn.geom_point(size=.5, show_legend=False)
        + pn.geom_text(pn.aes(label='elo.astype(int)'), labs, show_legend=False, size=6, nudge_y=+.2)
        + pn.scale_color_cmap(limits=(-1500, 0))
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_y_continuous(trans='log10')
        + pn.labs(
            x='Train-time compute (FLOPS-seconds)',
            y='Test-time compute (FLOPS-seconds)')
        + plot.IEEE())

def plot_calibrations():
    params = data.sample_calibrations()
    return (pn.ggplot(params, pn.aes(xmin='boardsize-.25', xmax='boardsize+.25', group='boardsize', fill='factor(boardsize)'))
        + pn.geom_hline(yintercept=.5, alpha=.2)
        + pn.geom_rect(pn.aes(ymin='lower', ymax='upper'), show_legend=False, color='k')
        + pn.geom_rect(pn.aes(ymin='mid', ymax='mid'), show_legend=False, color='k', size=2)
        + pn.scale_y_continuous(labels=percent_format())
        + pn.scale_fill_hue(l=.4)
        + pn.coord_cartesian(ylim=(.4, .6))
        + pn.labs(
            y='Win rate v. perfect play',
            x='Board size')
        + plot.IEEE())

def plot_optimal_model_size(ags):
    from statsmodels.formula import api as smf

    results = {}
    for b, g in ags.groupby('boardsize'):
        ordered = g.sort_values('elo').copy()
        ordered['params'] = g.width**2 * g.depth

        left = np.log10(g.train_flops.min())
        right = np.log10(g.train_flops.max())
        for f in np.linspace(left, right, 11)[1:]:
            subset = ordered[ordered.train_flops <= 10**f]
            results[b, 10**f] = subset.params.iloc[-1]
    df = pd.Series(results).reset_index()
    df.columns = ['boardsize', 'approx_flops', 'params']

    model = smf.ols('np.log10(params) ~ np.log10(approx_flops) + 1', df).fit()

    left, right = np.log10(df.approx_flops.min()), np.log10(df.approx_flops.max())
    preds = pd.DataFrame({'approx_flops': 10**np.linspace(left, right, 21)})
    preds['params'] = 10**model.predict(preds)

    labs = df.sort_values('approx_flops').groupby('boardsize').last().reset_index()
    labs['params'] = labs.apply(lambda r: df[df.approx_flops <= r.approx_flops].params.max(), axis=1)

    points = df.sort_values('approx_flops').groupby('boardsize').last().reset_index()

    return (pn.ggplot(df, pn.aes(x='approx_flops', y='params'))
        + pn.geom_line(pn.aes(color='factor(boardsize)', group='boardsize'), show_legend=False)
        + pn.geom_line(data=preds, linetype='dashed', size=.25)
        + pn.geom_point(pn.aes(color='factor(boardsize)', group='boardsize'), data=points, size=.5, show_legend=False)
        + pn.geom_text(pn.aes(color='factor(boardsize)', group='boardsize', label='boardsize'), data=labs, nudge_y=+.5, show_legend=False, size=6)
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_y_continuous(trans='log10')
        + pn.scale_color_hue(l=.4)
        + pn.labs(
            x='Train-time compute (FLOPS-seconds)',
            y='Optimal model size (params)')
        + plot.IEEE())




def hyperparams_table():
    s = pd.Series({
        'Number of envs': '32k',
        'Batch size': '32k',
        'Buffer size': '2m samples',
        'Learning rate': '1e-3',
        'MCTS node count': 64,
        r'MCTS $c_\text{puct}$': r'$\sfrac{1}{16}$',
        'MCTS noise $\epsilon$': r'$\sfrac{1}{4}$'})
    return s.to_latex(index=True, label='hyperparams', caption='Hyperparameters', escape=False, header=False)

def boardsize_hyperparams_table(ags):
    return (ags
        .groupby('boardsize')
        [['width', 'depth', 'samples', 'train_flops']]
        .max()
        .assign(train_flops=lambda df: df.train_flops.apply(lambda s: f'{s:.1G}'))
        .assign(samples=lambda df: df.samples.apply(lambda s: f'{s:.1G}'))
        .reset_index()
        .rename(columns={'boardsize': 'Board Size', 'width': 'Neurons', 'depth': 'Layers', 'samples': 'Samples', 'train_flops': 'Compute'})
        .to_latex(index=False, label='boardsize', caption='Board size-dependent hyperparameter limits'))

def parameters_table(ags, caption='Fitted Frontier Parameters', label='parameters'):
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
            .to_latex(index=True, label=label, caption=caption, escape=False, position='t'))

def direct_params_table(ags):
    ags = ags.copy()
    ags['elo'] = pd.concat([best.best_rates(best.TOPS[b]) for b in best.TOPS]).best_elo
    return parameters_table(ags, 'Fitted Frontier Parameters (Top-Agent Evaluation)', 'direct-parameters')

if __name__ == '__main__':
    ags = data.load()
    #TODO: Push this back into the database
    ags = data.with_times(ags)
    upload(plot_hex)
    upload(plot_flops_curves, ags)
    upload(plot_frontiers, ags)
    upload(plot_direct_frontiers, ags)
    upload(plot_resid_var, ags)
    upload(plot_runtimes, ags)
    upload(plot_train_test, ags)
    upload(plot_elos)
    upload(plot_test, ags)
    upload(plot_calibrations)

    overleaf.table(boardsize_hyperparams_table(ags), 'boardsize_hyperparams')
    overleaf.table(parameters_table(ags), 'parameters')
    overleaf.table(direct_params_table(ags), 'direct-parameters')
    overleaf.table(hyperparams_table(), 'hyperparams')