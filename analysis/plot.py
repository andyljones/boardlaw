import plotnine as pn

def mpl_theme(width=12, height=8):
    return [
        pn.theme_matplotlib(),
        pn.theme(
            figure_size=(width, height), 
            strip_background=pn.element_rect(color='w', fill='w'),
            panel_grid=pn.element_line(color='k', alpha=.1))
            ]

def poster_sizes():
    return pn.theme(
        text=pn.element_text(size=18),
        title=pn.element_text(size=18),
        legend_title=pn.element_text(size=18))

class IEEE(pn.theme):

    def __init__(self):
        super().__init__(complete=True)

        self._rcParams.update({
            'figure.figsize': (5, 4),
            'font.family': 'serif',
            'font.size': 8,
            'axes.grid': True,
            'axes.grid.which': 'both',
            'grid.alpha': .2})
        

def ieee():
    # https://github.com/garrettj403/SciencePlots/blob/master/styles/journals/ieee.mplstyle
    return [
        pn.theme_mpl(),
        pn.theme(
            strip_background=pn.element_rect(color='w', fill='w'),
            panel_grid=pn.element_line(color='k', alpha=.1, size=.25),

            text=pn.element_text(family='serif', size=8),
            legend_title=pn.element_text(size=8),
            figure_size=(3.3, 2.5), 
            dpi=180,)]

def no_colorbar_ticks():
    return pn.guides(color=pn.guide_colorbar(ticks=False))