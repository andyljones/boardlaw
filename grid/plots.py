import pandas as pd
from plotnine import *
from . import data

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
        + theme_matplotlib()
        + guides(
            color=guide_colorbar(ticks=False))
        + theme(
            figure_size=(18, 6), 
            strip_background=element_rect(color='w', fill='w'),
            panel_grid=element_line(color='k', alpha=.1)))
