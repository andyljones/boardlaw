import numpy as np
from . import plot, data
import plotnine as pn

def plot_frontiers(ags):
    df = data.modelled_elos(ags)
    labels = df.sort_values('train_flops').groupby('boardsize').first().reset_index()

    return (pn.ggplot(df, pn.aes(x='train_flops/(1e12*3600)', color='factor(boardsize)', group='boardsize'))
                + pn.geom_line(pn.aes(y='400/np.log(10)*elo'), size=.5, show_legend=False)
                + pn.geom_line(pn.aes(y='400/np.log(10)*elohat'), size=.25, linetype='dashed', show_legend=False)
                + pn.geom_text(pn.aes(y='400/np.log(10)*elohat', label='boardsize'), data=labels, show_legend=False, size=8, nudge_x=-.25, nudge_y=-25)
                + pn.labs(
                    x='GPU-hours', 
                    y='Elo v. perfect play')
                + pn.scale_x_continuous(trans='log10')
                + pn.coord_cartesian(None, (None, 0))
                + plot.ieee())


