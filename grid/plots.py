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

def plot_sample_eff(df):
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
        plots.mpl_theme(18, 12))

def plot_convergence_rate(df):
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