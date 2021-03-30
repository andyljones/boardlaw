import cvxpy as cp
import pandas as pd
import numpy as np
from . import elos, sql, storage
from tqdm.auto import tqdm

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

def nash_solns(ags, ws, gs):
    [boardsize] = ags.boardsize.unique()
    schedule = storage.savepoints(boardsize)
    assignment = abs(np.log10(ags.train_flops.reindex(ws.index).values)[:, None] - np.log10(schedule)[None, :]).argmin(1)

    rate = (ws/gs).fillna(.5)

    strats = {}
    payoffs = {}
    for i in range(len(schedule)):
        for j in range(len(schedule)):
            iflops, jflops = schedule[i], schedule[j]
            A = rate.loc[assignment == i].loc[:, assignment == j]
            if (i != j) and (A.size > 0):
                x, y = equilibrium(A)
                strats[iflops, jflops] = x, y 
                payoffs[iflops, jflops] = x @ A @ y
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


def nash_payoffs(boardsize=None):
    if boardsize is None:
        return pd.concat({b: nash_payoffs(b) for b in tqdm(range(3, 10))}, names=('boardsize',)).reset_index(level=0).reset_index(drop=True)
    ags = (sql.agent_query()
            .query('test_nodes == 64')
            .loc[lambda df: df.description == f'bee/{boardsize}'])
    trials = (sql
                .trial_query(boardsize)
                .loc[lambda df: df.black_agent.isin(ags.index) & df.white_agent.isin(ags.index)])

    ws, gs = elos.symmetrize(trials)
    strats, payoffs = nash_solns(ags, ws, gs)

    payoffs = payoffs.stack().reset_index()
    payoffs.columns = ['strategy', 'challenger', 'winrate']

    return payoffs

def nash_grads(payoffs):
    grads = {}
    for boardsize, g in payoffs.groupby('boardsize'):
        df = g.pivot('strategy', 'challenger', 'winrate')
        mid_up = np.log10(df.index)[1:]
        mid_down = np.log10(df.index)[:-1]
        dy = np.diag(df, -1) - np.diag(df, +1)
        dx = mid_up - mid_down
        grads[boardsize] = pd.Series(dy/dx, 10**((mid_up + mid_down)/2))
    grads = pd.concat(grads, names=('boardsize',)).rename('grad').reset_index()
    return grads