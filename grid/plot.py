import pandas as pd
import numpy as np
import plotnine as pn
from . import data

def mpl_theme(width=12, height=8):
    return [
        pn.theme_matplotlib(),
        pn.theme(
            figure_size=(width, height), 
            strip_background=pn.element_rect(color='w', fill='w'),
            panel_grid=pn.element_line(color='k', alpha=.1))]

def poster_sizes():
    return pn.theme(text=pn.element_text(size=18),
                title=pn.element_text(size=18),
                legend_title=pn.element_text(size=18))

def plot_flops(snaps):
    return (pn.ggplot(snaps, pn.aes(x='flops', y='μ', group='run', color='factor(boardsize)'))
        + pn.geom_line()
        + pn.geom_point(size=.5)
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_discrete(name='boardsize')
        + mpl_theme()
        + poster_sizes())

def _select_frontier(g, var):
    ordered = g.groupby(var).μ.max().sort_index()
    maxes = ordered.expanding().max()
    return ordered[ordered == maxes]

def plot_data_frontier(snaps, var='params'):
    df = (snaps
            .groupby(['boardsize'])
            .apply(_select_frontier, var)
            .reset_index())
    return (pn.ggplot(df, pn.aes(x=var, y='400/np.log(10)*μ', color='factor(boardsize)', group='boardsize'))
        + pn.geom_line(size=2)
        + pn.labs(
            x='training flops', 
            y='elo v. perfect play')
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_discrete(name='boardsize')
        + mpl_theme()
        + poster_sizes())

def plot_flops_frontier(snaps):
    return (plot_data_frontier(snaps, 'flops')
        + pn.labs(title='performance frontier in terms of compute'))

def _interp_frontier(g, var):
    xl, xr = g[var].pipe(np.log10).min(), g[var].pipe(np.log10).max()
    xs = np.linspace(xl, xr, 101)
    ys = {}
    for run, gg in g.groupby('run'):
        xp = gg[var].pipe(np.log10).values
        yp = gg.μ.values
        ys[run] = np.interp(xs, xp, yp, np.nan, np.nan)
    ys = pd.DataFrame(ys, index=10**xs)

    return ys.max(1).pipe(lambda s: s[s.expanding().max() == s]).rename_axis(index='flops').rename('μ')

def plot_interp_frontier(snaps):
    df = (snaps
            .groupby('boardsize')
            .apply(_interp_frontier, 'flops')
            .reset_index())
    return (pn.ggplot(df, pn.aes(x='flops', y='400/np.log(10)*μ', color='factor(boardsize)', group='boardsize'))
            + pn.geom_line(size=2)
            + pn.labs(
                x='training flops', 
                y='elo v. perfect play',
                title='performance frontier in terms of compute')
            + pn.scale_x_continuous(trans='log10')
            + pn.scale_color_discrete(name='boardsize')
            + mpl_theme()
            + poster_sizes())

def elo_errors(snaps):
    errs = []
    for b in snaps.boardsize.unique():
        μ = snaps.loc[lambda df: df.boardsize == b].μ
        games, wins = data.load(b)

        rates = (wins/games).reindex(index=μ.index, columns=μ.index)

        diffs = pd.DataFrame(μ.values[:, None] - μ.values[None, :], μ.index, μ.index)
        expected = 1/(1 + np.exp(-diffs))

        errs.append((rates - expected).abs().mean())
    return pd.concat(errs)

def plot_elo_errors(snaps):
    snaps['err'] = elo_errors(snaps)
    return (pn.ggplot(snaps)
        + pn.geom_point(pn.aes(x='μ', y='err', group='run', color='flops'), size=.5)
        + pn.scale_color_continuous(trans='log10')
        + pn.facet_wrap('boardsize', labeller='label_both')
        + pn.guides(color=pn.guide_colorbar(ticks=False))
        + mpl_theme(18, 12)
        + poster_sizes())
