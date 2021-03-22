import plotnine as pn

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


def ieee():
    # https://github.com/garrettj403/SciencePlots/blob/master/styles/journals/ieee.mplstyle
    return pn.theme(
        text=pn.element_text(family='serif')
    )
# class IEEE(pn.theme.theme):

#     def __init__(self):
#         super().__init__()
#         self._rcParams[]

def no_colorbar_ticks():
    return pn.guides(color=pn.guide_colorbar(ticks=False))