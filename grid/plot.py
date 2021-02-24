import statsmodels.formula.api as smf
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
    cutoffs = {
        3: (6e10, 2e11),
        4: (2e11, 2e12),
        5: (3e11, 2e13),
        6: (7e11, 3e14),
        7: (1e12, 3e15),
        8: (1e12, 1e16),
        9: (1e12, 1e17)}
    cutoffs = pd.DataFrame.from_dict(cutoffs, orient='index', columns=('lower', 'upper')).rename_axis(index='boardsize')

    df = (snaps
            .join(cutoffs, on='boardsize')
            .query('lower <= flops <= upper')
            .groupby('boardsize')
            .apply(_interp_frontier, 'flops')
            .reset_index())

    model = smf.ols('μ ~ boardsize + np.log10(flops) + 1', df).fit()
    df['μhat'] = model.predict(df)

    ps = model.params.mul(400/np.log(10)).apply(lambda x: f'{float(f"{x:.1g}"):g}')
    s = f'$\mathrm{{elo}} = {ps.boardsize} \cdot \mathrm{{boardsize}} + {ps["np.log10(flops)"]} \cdot \ln_{{10}}(\mathrm{{flops}}) + C$'

    (pn.ggplot(df, pn.aes(x='flops', color='factor(boardsize)', group='boardsize'))
            + pn.geom_line(pn.aes(y='400/np.log(10)*μ'), size=2)
            + pn.geom_line(pn.aes(y='400/np.log(10)*μhat'), size=1, linetype='dashed')
            + pn.annotate('text', 5e14, -2500, label=s, size=20)
            + pn.labs(
                x='training flops', 
                y='elo v. perfect play',
                title='performance frontier in terms of compute')
            + pn.scale_x_continuous(trans='log10')
            + pn.scale_color_discrete(name='boardsize')
            + pn.coord_cartesian(None, (None, 0))
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
