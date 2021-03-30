import cvxpy as cp
import pandas as pd
import numpy as np
from . import elos, sql

def equilibrium(A):
    # https://www.cs.cmu.edu/~ggordon/780-spring09/slides/Game%20theory%20lecture2-algs%20for%20normal%20form.pdf
    x = cp.Variable(A.shape[0])
    z = cp.Variable()

    bounds = [
        z - A.values.T @ x <= 0, 
        cp.sum(x) == 1, 
        x >= 0]

    cp.Problem(cp.Maximize(z), bounds).solve()

    x = pd.Series(x.value, A.index)
    y = pd.Series(bounds[0].dual_value, A.columns)
    return x, y

def nash_solns(snaps, ws, gs):
    rate = (ws/gs).fillna(.5)

    strats = {}
    payoffs = {}
    for i, i_group in snaps.reindex(rate.index).groupby('idx'):
        for j, j_group in snaps.reindex(rate.index).groupby('idx'):
            if i != j:
                A = rate.reindex(index=i_group.index, columns=j_group.index)
                x, y = equilibrium(A)
                strats[i, j] = x, y 
                payoffs[i, j] = x @ A @ y
    payoffs = pd.Series(payoffs).unstack()
    return strats, payoffs

def nash_elos(payoffs):
    ws = 1000*payoffs.fillna(.5)
    gs = 0*ws + 1000

    return pd.Series(elos.solve(ws, gs), payoffs.index)

def plot_nash_elos(joint):
    import plotnine as pn
    from . import plot
    import statsmodels.formula.api as smf

    cutoffs = {
        3: (5e10, 2e11),
        4: (1e11, 2e12),
        5: (2e11, 8e12),
        6: (7e11, 2e14),
        7: (1e12, 3e15),
        8: (3e12, 1e16),
        9: (6e12, 1e17)}
    cutoffs = pd.DataFrame.from_dict(cutoffs, orient='index', columns=('lower', 'upper')).rename_axis(index='boardsize')

    df =(
        joint
        .join(cutoffs, on='boardsize')
        .query('lower <= flops <= upper'))

    model = smf.ols('elo ~ boardsize + np.log10(flops) + 1', df).fit()
    df['elohat'] = model.predict(df)

    ps = model.params.mul(400/np.log(10)).apply(lambda x: f'{float(f"{x:.2g}"):g}')
    s = f'$\mathrm{{elo}} = {ps.boardsize} \cdot \mathrm{{boardsize}} + {ps["np.log10(flops)"]} \cdot \ln_{{10}}(\mathrm{{flops}}) + C$'

    return (pn.ggplot(df)
        + pn.geom_line(pn.aes(x='flops', y='400/np.log(10)*elo', group='boardsize', color='factor(boardsize)'), size=2)
        + pn.geom_line(pn.aes(x='flops', y='400/np.log(10)*elohat', color='factor(boardsize)'), size=1, linetype='dashed')
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_discrete(name='boardsize')
        + pn.labs(title='elos of nash equilibiria')
        + pn.annotate('text', 3e14, -2000, label=s, size=20)
        + pn.coord_cartesian(None, (None, 0))
        + plot.mpl_theme()
        + plot.poster_sizes())


def run_nash_elos(boardsize):
    ags = (sql.agent_query()
            .query('test_nodes == 64')
            .loc[lambda df: df.description == f'bee/{boardsize}'])
    trials = (sql
                .trial_query(boardsize)
                .loc[lambda df: df.black_agent.isin(ags.index) & df.white_agent.isin(ags.index)])

    ws, gs = elos.symmetrize(trials)
    strats, payoffs = nash_solns(ags, ws, gs)
    elos = nash_elos(payoffs)

    return elos
