import numpy as np
from . import plot, data
import plotnine as pn
import matplotlib.patheffects as path_effects
from boardlaw import arena, analysis

RUNS = {
    3: ('2021-02-17 21-01-19 arctic-ease', 20),
    9: ('2021-02-20 23-35-25 simple-market', 20)}

def plot_termination(n_envs=1, boardsize=9):
    run, idx = RUNS[boardsize]

    world = arena.common.worlds(run, n_envs)
    agent = arena.common.agent(run, idx)
    trace = analysis.rollout(world, [agent, agent], n_trajs=1)

    penult = trace.worlds[-2]
    actions = trace.actions[-1]
    ult, _ = penult.step(actions, reset=False)

    return ult.display()

def plot_frontiers(ags):
    df = data.modelled_elos(ags)
    labels = df.sort_values('train_flops').groupby('boardsize').first().reset_index()

    return (pn.ggplot(df, pn.aes(x='train_flops', color='factor(boardsize)', group='boardsize'))
                + pn.geom_line(pn.aes(y='400/np.log(10)*elo'), size=.5, show_legend=False)
                + pn.geom_line(pn.aes(y='400/np.log(10)*elohat'), size=.25, linetype='dashed', show_legend=False)
                + pn.geom_text(pn.aes(y='400/np.log(10)*elohat', label='boardsize'), data=labels, show_legend=False, size=6, nudge_x=-.25, nudge_y=-15)
                + pn.labs(
                    x='TFLOPS', 
                    y='Elo v. perfect play')
                + pn.scale_color_discrete(l=.4)
                + pn.scale_x_continuous(trans='log10')
                + pn.coord_cartesian(None, (None, 0))
                + plot.IEEE())


