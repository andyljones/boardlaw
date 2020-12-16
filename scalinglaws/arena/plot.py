import activelo
import numpy as np
from rebar import dotdict
import pandas as pd
from . import database
import seaborn as sns
import matplotlib.pyplot as plt
from rebar import paths

def to_pandas(soln, games):
    return dotdict.dotdict(
        μ=pd.Series(soln.μ, games.index),
        Σ=pd.DataFrame(soln.Σ, games.index, games.index))

def condition(soln, name):
    μ, Σ = soln.μ, soln.Σ 
    σ2 = np.diag(Σ) + Σ.loc[name, name] - 2*Σ[name]
    μc = μ - μ[name]
    return μc.drop(name), σ2.drop(name)**.5

def drop_names(df, names):
    return df.loc[
        ~df.index.isin(names), 
        ~df.columns.isin(names)]

def drop_latest(df):
    return df.loc[
        ~df.index.str.endswith('latest'), 
        ~df.columns.str.endswith('latest')]

def periodic(run_name, target=None, drop=[]):
    run_name = paths.resolve(run_name)
    games, wins = database.symmetric_pandas(run_name)
    games, wins = drop_latest(games), drop_latest(wins)
    games, wins = drop_names(games, drop), drop_names(wins, drop)
    soln = activelo.solve(games.values, wins.values)

    soln = to_pandas(soln, games)
    if target == 'mohex':
        μ, σ = condition(soln, 'mohex')
        title = f'{run_name}: eElo v. mohex'
    elif target == 'first':
        μ, σ = condition(soln, soln.μ.index[0])
        title = f'{run_name}: eElo v. first'
    else:
        μ, σ = soln.μ, np.diag(soln.Σ)**.5
        title = f'{run_name} eElo, raw'

    fig, axes = plt.subplots(1, 1, squeeze=False)

    ax = axes[0, 0]
    ax.errorbar(np.arange(len(μ)), μ, yerr=σ, marker='.', capsize=2, linestyle='')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(μ)))
    ax.set_xticklabels(μ.index, rotation=-90)
    ax.grid(True, axis='y')

    return μ

def heatmap(run_name=-1, drop=[]):
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
    from scalinglaws.arena import database
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