import numpy as np
import pandas as pd
from boardlaw import arena
from grid import data, eval, plot
import activelo
import statsmodels.formula.api as smf
import plotnine as pn

def evaluate(snap):
    A = arena.common.agent(snap.run, snap.idx, 'cuda')
    B = arena.common.agent(snap.run, snap.idx, 'cuda')
    worlds = arena.common.worlds(snap.run, 1024, 'cuda')

    A.kwargs = {'n_nodes': 2}
    results = []
    for n_nodes in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        print(f'Evaluating {n_nodes}')
        B.kwargs = {'n_nodes': n_nodes}
        results.extend(arena.common.evaluate(worlds, [(2, A), (n_nodes, B)]))

    agents = sorted({n for r in results for n in r.names})

    games  = pd.DataFrame(index=agents, columns=agents).fillna(0).astype(int)
    wins  = pd.DataFrame(index=agents, columns=agents).fillna(0).astype(int)
    games, wins = eval.update(games, wins, results)

    return games, wins

def fit_predict(games, wins):
    soln = activelo.solve(games, wins)
    elos = (soln.μ - soln.μ.iloc[0]).mul(400/np.log(10))

    df = elos.rename_axis(index='nodes').rename('elo').reset_index()

    model = smf.ols('elo ~ np.log10(nodes) + 1', df).fit()
    df['elohat'] = model.predict(df)

    return df, model

def plot_elos(df, model):
    ps = model.params.apply(lambda x: f'{float(f"{x:.2g}"):g}')
    s = f'$\mathrm{{elo}} = {ps["np.log10(nodes)"]} \cdot \ln_{{10}}(\mathrm{{nodes}}) + C$'

    return (pn.ggplot(df, pn.aes(x='nodes'))
        + pn.geom_line(pn.aes(y='elo'))
        + pn.geom_point(pn.aes(y='elo'))
        + pn.geom_line(pn.aes(y='elohat'), linetype='dashed')
        + pn.scale_x_continuous(trans='log2')
        + pn.annotate('text', 100, 0, label=s, size=20)
        + pn.labs(
            x='number of MCTS nodes', 
            y='elo v. MCTS with 2 nodes')
        + plot.mpl_theme()
        + plot.poster_sizes())

def run():
    snaps = data.snapshot_solns(7)

    snap = (snaps
                    .loc[lambda df: df.flops <= 1e14]
                    .sort_values('μ')
                    .iloc[-1])

    games, wins = evaluate(snap)

    df, model = fit_predict(games, wins)

    return plot_elos(df, model)