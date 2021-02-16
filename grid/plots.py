import pandas as pd
from plotnine import *
from . import data

def mpl_theme(width=12, height=8):
    return [
        theme_matplotlib(),
        guides(
            color=guide_colorbar(ticks=False)),
        theme(
            figure_size=(width, height), 
            strip_background=element_rect(color='w', fill='w'),
            panel_grid=element_line(color='k', alpha=.1))]

def plot_sigmoids(aug):
    return (ggplot(data=aug, mapping=aes(x='width', y='rel_elo', color='depth'))
        + geom_line(mapping=aes(group='depth'))
        + geom_point()
        + facet_wrap('boardsize', nrow=1)
        + scale_x_continuous(trans='log2')
        + scale_color_continuous(trans='log2')
        + coord_cartesian((-.1, None), (0, 1), expand=False)
        + labs(
            title='larger boards lead to slower scaling',
            y='normalised elo (entirely random through to perfect play)')
        + mpl_theme(18, 6))

def plot_sample_eff():
    df = data.load()

    return (ggplot(
        data=df
            .iloc[5:]
            .rolling(15, 1).mean()
            .unstack().unstack(0)
            .loc[7]
            .reset_index()
            .dropna()) + 
        geom_line(aes(x='np.log10(samples)', y='elo/9.03 + 1', group='depth', color='np.log2(depth)')) + 
        labs(title='sample efficiency forms a large part of the advantage of depth (7x7)') + 
        facet_wrap('width') +
        mpl_theme(18, 12))

def plot_convergence_rate(df):
    df = data.load()

    diffs = {}
    for b, t in data.TAILS.items():
        live = df.elo[b].dropna(0, 'all')
        diffs[b] = (live - live.iloc[-1])/data.min_elos().abs()[b]
    diffs = pd.concat(diffs, 1)

    (ggplot(
        data=(
            diffs
                .unstack()
                .rename('elo')
                .reset_index()
                .rename(columns={'level_0': 'boardsize'})
                .dropna()
                .assign(
                    s=lambda df: df._time.astype(int)/1e9,
                    g=lambda df: df.depth.astype(str) + df.width.astype(str)))) + 
        geom_line(aes(x='_time', y='elo', group='g', color='np.log2(width)')) + 
        facet_wrap('boardsize', scales='free') +
        plots.mpl_theme() + 
        labs(title='runs converge much faster than I thought') + 
        theme(panel_spacing_y=.3, panel_spacing_x=.5))

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
    return (ggplot(
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
            geom_line(aes(x='flops', y='norm_elo', color='params', group='g')) + 
            labs(
                x='training FLOPS',
                y='normalized elo; 0 is random, 1 is perfect play',
                title='alphazero performance increases as a sigmoid in compute;\nincreasing boardsize by 2 means a ~100x increase in the compute needed for perfect play') +
            scale_x_continuous(trans='log10') + 
            scale_color_continuous(trans='log10') + 
            facet_wrap('boardsize', labeller='label_both') +
            coord_cartesian(None, (0, 1)) +
            mpl_theme(18, 15) +
            theme(
                text=element_text(size=18),
                title=element_text(size=24),
                legend_title=element_text(size=18)))

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