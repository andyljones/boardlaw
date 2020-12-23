import activelo
import numpy as np
from rebar import dotdict
import pandas as pd
from . import database, analysis
import matplotlib.pyplot as plt
from rebar import paths
import copy

def periodic(run_name=-1, target=None, filter=''):
    run_name = paths.resolve(run_name)
    elos = analysis.elos(run_name, target, filter=filter)
    if target:
        title = f'{run_name} eElo v. {target}'
    else:
        title = f'{run_name} eElo, raw'

    fig, axes = plt.subplots(1, 1, squeeze=False)

    ax = axes[0, 0]
    ax.errorbar(np.arange(len(elos)), elos.μ, yerr=elos.σ, marker='.', capsize=2, linestyle='')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(elos.μ)))
    ax.set_xticklabels(elos.μ.index, rotation=-90)
    ax.grid(True, axis='y')

    return elos.μ

def heatmap(run_name=-1, drop=[]):
    import seaborn as sns

    rates = database.symmetric_wins(run_name)/database.symmetric_games(run_name)
    rates = (rates
        .drop(drop,  axis=0)
        .drop(drop, axis=1)
        .rename(index=lambda c: c[7:-9], columns=lambda c: c[7:-9]))
    rates.values[np.diag_indices_from(rates)] = .5
    # rates.values[np.triu_indices_from(rates)] = np.nan
    rates.index.name = 'agent'
    rates.columns.name = 'challenger'
    ax = sns.heatmap(rates, cmap='RdBu', vmin=0, vmax=1, square=True)
    ax.set_facecolor('dimgrey')
    ax.set_title(f'{paths.resolve(run_name)} winrate')

def nontransitivities(run_name=-1):
    import seaborn as sns
    from boardlaw.arena import database
    w, n = database.symmetric_wins(run_name), database.symmetric_games(run_name)
    r = w/n
    e = (r*(1-r)/n)**.5

    conf = (r > .5 + 2*e).astype(float) - (r < .5 - 2*e).astype(float)

    C = conf.values
    results = np.zeros_like(C)
    N = len(results)
    for i in range(N):
        for j in range(N):
            bad = False
            for k in range(N):
                if (C[i, j] == +1) and (C[i, k] == -1) and (C[j, k] == +1):
                    bad = True
                if (C[i, j] == -1) and (C[i, k] == +1) and (C[j, k] == -1):
                    bad = True
            results[i, j] = bad
    results = pd.DataFrame(results, 
        [c[7:-9] for c in conf.index], 
        [c[7:-9] for c in conf.columns])

    sns.heatmap(results, cmap='Greens', square=True, vmax=1, vmin=0)

    return results

def errors(run_name=-1, filter='.*'):
    run_name = paths.resolve(run_name)
    games, wins = database.symmetric_pandas(run_name)
    games, wins = analysis.mask(games, wins, filter)
    soln = activelo.solve(games.values, wins.values)

    rates = wins/games

    expected = 1/(1 + np.exp(-soln.μ[:, None] + soln.μ[None, :]))
    actual = rates.where(games > 0, np.nan).values

    resid_var = np.nanmean((actual - expected)**2)/np.nanmean(actual**2)
    corr = np.corrcoef(actual[~np.isnan(actual)], expected[~np.isnan(actual)])[0, 1]

    fig = plt.figure()
    gs = plt.GridSpec(4, 3, fig, height_ratios=[20, 1, 20, 1])
    fig.set_size_inches(18, 12)

    # Top row
    cmap = copy.copy(plt.cm.RdBu)
    cmap.set_bad('lightgrey')
    kwargs = dict(cmap=cmap, vmin=0, vmax=1, aspect=1)

    ax = plt.subplot(gs[0, 0])
    ax.imshow(actual, **kwargs)
    ax.set_title('actual')

    ax = plt.subplot(gs[0, 1])
    im = ax.imshow(expected, **kwargs)
    ax.set_title('expected')

    ax = plt.subplot(gs[1, :2])
    plt.colorbar(im, cax=ax, orientation='horizontal')

    # Top right
    ax = plt.subplot(gs[0, 2])
    elos = analysis.elos(run_name, target=0)
    ax.errorbar(np.arange(len(elos)), elos.μ, yerr=elos.σ, marker='.', capsize=2, linestyle='')
    ax.set_title('elos v. first')
    ax.grid()


    # Bottom left
    ax = plt.subplot(gs[2, 0])
    im = ax.imshow(actual - expected, vmin=-1, vmax=+1, aspect=1, cmap='RdBu')
    ax.set_title('error')

    ax = plt.subplot(gs[3, 0])
    plt.colorbar(im, cax=ax, orientation='horizontal')
    # ax.annotate(f'resid var: {resid_var:.0%}, corr: {corr:.0%}', (.5, -1.2), ha='center', xycoords='axes fraction')

    # Bottom right
    ax = plt.subplot(gs[2, 1])
    se = (expected*(1-expected)/games)**.5
    im = ax.imshow((actual - expected)/se, vmin=-3, vmax=+3, aspect=1, cmap='RdBu')
    ax.set_title('standard error')

    ax = plt.subplot(gs[3, 1])
    plt.colorbar(im, cax=ax, orientation='horizontal')
    # ax.annotate(f'resid var: {resid_var:.0%}, corr: {corr:.0%}', (.5, -1.2), ha='center', xycoords='axes fraction')




