import pandas as pd
import plotnine as pn
from . import data, refine
import numpy as np
import matplotlib.pyplot as plt

def mpl_theme(width=12, height=8):
    return [
        pn.theme_matplotlib(),
        pn.guides(
            color=pn.guide_colorbar(ticks=False)),
        pn.theme(
            figure_size=(width, height), 
            strip_background=pn.element_rect(color='w', fill='w'),
            panel_grid=pn.element_line(color='k', alpha=.1))]

def poster_sizes():
    return pn.theme(text=pn.element_text(size=18),
                title=pn.element_text(size=18),
                legend_title=pn.element_text(size=18))

def plot_sigmoids(aug):
    return (pn.ggplot(data=aug, mapping=pn.aes(x='width', y='rel_elo', color='depth'))
        + pn.geom_line(mapping=pn.aes(group='depth'))
        + pn.geom_point()
        + pn.facet_wrap('boardsize', nrow=1)
        + pn.scale_x_continuous(trans='log2')
        + pn.scale_color_continuous(trans='log2')
        + pn.coord_cartesian((-.1, None), (0, 1), expand=False)
        + pn.labs(
            title='larger boards lead to slower scaling',
            y='normalised elo (entirely random through to perfect play)')
        + mpl_theme(18, 6))

def plot_sample_eff():
    df = data.load()

    return (pn.ggplot(
        data=df
            .iloc[5:]
            .rolling(15, 1).mean()
            .unstack().unstack(0)
            .loc[7]
            .reset_index()
            .dropna()) + 
        pn.geom_line(pn.aes(x='np.log10(samples)', y='elo/9.03 + 1', group='depth', color='np.log2(depth)')) + 
        pn.labs(title='sample efficiency forms a large part of the advantage of depth (7x7)') + 
        pn.facet_wrap('width') +
        mpl_theme(18, 12))

def plot_convergence_rate(df):
    df = data.load()

    diffs = {}
    for b, t in data.TAILS.items():
        live = df.elo[b].dropna(0, 'all')
        diffs[b] = (live - live.iloc[-1])/data.min_elos().abs()[b]
    diffs = pd.concat(diffs, 1)

    (pn.ggplot(
        data=(diffs
                .unstack()
                .rename('elo')
                .reset_index()
                .rename(columns={'level_0': 'boardsize'})
                .dropna()
                .assign(
                    s=lambda df: df._time.astype(int)/1e9,
                    g=lambda df: df.depth.astype(str) + df.width.astype(str)))) + 
        pn.geom_line(pn.aes(x='_time', y='elo', group='g', color='np.log2(width)')) + 
        pn.facet_wrap('boardsize', scales='free') +
        mpl_theme() + 
        pn.labs(title='runs converge much faster than I thought') + 
        pn.theme(panel_spacing_y=.3, panel_spacing_x=.5))

def flops(df):
    intake = (df.boardsize**2 + 1)*df.width
    body = (df.width**2 + df.width) * df.depth
    output = df.boardsize**2 * (df.width + 1)
    return 64*df.samples*(intake + body + output)

def params(df):
    intake = (df.boardsize**2 + 1)*df.width
    body = (df.width**2 + df.width) * df.depth
    output = df.boardsize**2 * (df.width + 1)
    return intake + body + output

def plot_compute_perf(df=None):
    df = data.load() if df is None else df
    return (pn.ggplot(
            data=df
                .iloc[5:]
                .pipe(lambda df: df.ewm(span=1).mean().where(df.bfill().notnull()))
                .unstack().unstack(0)
                .reset_index()
                .assign(params=params)
                .assign(flops=flops)
                .assign(g=lambda df: df.width.astype(str)+df.depth.astype(str))
                .assign(norm_elo=data.normalised_elo)
                .dropna()) + 
            pn.geom_line(pn.aes(x='flops', y='norm_elo', color='params', group='g')) + 
            pn.labs(
                x='training FLOPS',
                y='normalized elo; 0 is random, 1 is perfect play',
                title='alphazero performance increases as a sigmoid in compute;\nincreasing boardsize by 2 means a ~100x increase in the compute needed for perfect play') +
            pn.scale_x_continuous(trans='log10') + 
            pn.scale_color_continuous(trans='log10') + 
            pn.facet_wrap('boardsize', labeller='label_both') +
            pn.coord_cartesian(None, (0, 1)) +
            mpl_theme(18, 15) +
            poster_sizes())

def plot_compute_perf_frontier():
    df = data.load()

    interp = df.interpolate().where(df.bfill().notnull())
    stacked = interp.stack([1, 2, 3]).reset_index().dropna()
    stacked['flops'] = flops(stacked)

    best = stacked.groupby(['_time', 'boardsize']).elo.idxmax()
    best = stacked.loc[best]

    index = 10**np.linspace(0, np.log10(stacked.flops.max()), 1001)
    regular = pd.pivot_table(stacked, 'elo', ('flops',), ('boardsize', 'width', 'depth')).ffill().reindex(index, method='nearest')
    frontier = regular.expanding().max().groupby(axis=1, level=0).max().div(data.min_elos(), axis=1).mul(-1).add(1)

    with plt.style.context('seaborn-poster'):
        slope = (frontier > .95).drop(11, axis=1).idxmax()
        ax = slope.plot(logy=True, marker='o', linestyle='--', grid=True)
        ax.set_ylabel('training FLOPS')
        ax.set_xlabel('boardsize')
        ax.set_title('FLOPS needed to hit 95% of perfect play is roughly linear')

def plot_frontier_slopes():
    df = data.load()

    interp = df.interpolate().where(df.bfill().notnull())
    stacked = interp.stack([1, 2, 3]).reset_index().dropna()
    stacked['flops'] = flops(stacked)
    stacked = stacked[stacked.elo_std < .5]

    best = stacked.groupby(['_time', 'boardsize']).elo.idxmax()
    best = stacked.loc[best]

    index = 10**np.linspace(0, np.log10(stacked.flops.max()), 1001)
    regular = pd.pivot_table(stacked, 'elo', ('flops',), ('boardsize', 'width', 'depth')).ffill().reindex(index, method='nearest')

    frontier = regular.expanding().max().groupby(axis=1, level=0).max()

    slope = frontier.div(-data.min_elos().drop(11), axis=1).add(1).gt(.95).idxmax()
    x = slope.reset_index()
    x.columns = ['boardsize', 'flops']

    y = frontier.stack().rename('elo').reset_index()
    y['shifted'] = y.flops.div(x.set_index('boardsize').flops[y.boardsize.values].values)

    (pn.ggplot(data=y)
        + pn.geom_line(pn.aes(x='shifted', y='elo', color='boardsize', group='boardsize'), size=2)
        + pn.scale_x_continuous(trans='log10')
        + pn.coord_cartesian((-4, 1))
        + pn.labs(
            x='FLOPS as a fraction of 95% of perfect play',
            title='slope of the frontier seems the same across boardsizes?')
        + mpl_theme())

def plot_refine_results():
    joint = refine.solutions()
    joint['g'] = joint.index.str[:-2]
    joint['flops'] = flops(joint)
    joint['params'] = params(joint)

    (pn.ggplot(data=joint)
        + pn.geom_point(pn.aes(x='flops', y='μ', group='g', color='params'))
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_continuous(trans='log10')
        + pn.facet_wrap('boardsize')
        + pn.labs(title='')
        + mpl_theme(12, 8)
        + poster_sizes())

def plot_refine_superimposed():
    joint = refine.solutions()
    joint['g'] = joint.index.str[:-2]
    joint['flops'] = flops(joint)
    joint['params'] = params(joint)

    (pn.ggplot(data=joint[joint.index.str[-2] != '0'])
        + pn.geom_point(pn.aes(x='flops', y='μ', group='g', color='boardsize'))
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_continuous(trans='log10')
        + pn.labs(title='slope is about the same in every board size')
        + mpl_theme(12, 8)
        + poster_sizes())

def plot_refine_arch_annotations():
    joint = refine.solutions()
    joint['g'] = joint.index.str[:-2]
    joint['flops'] = flops(joint)
    joint['params'] = params(joint)
    joint['arch'] = joint.depth.astype(str) + '/' + joint.depth.astype(str)
    joint['idx'] = joint.index.str.extract('.*D(.*)S', expand=False)

    (pn.ggplot(data=(joint
                        .loc[lambda df: (df.boardsize == 9)]))
        + pn.geom_text(pn.aes(x='flops', y='μ', label='idx'), size=8)
        + pn.scale_x_continuous(trans='log10')
        + pn.labs(title='')
        + mpl_theme(12, 8)
        + poster_sizes())