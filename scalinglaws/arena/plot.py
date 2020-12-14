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

def heatmap(run_name=-1):
    rates = database.symmetric_wins(run_name)/database.symmetric_games(run_name)
    sns.heatmap(rates, cmap='RdBu', vmin=0, vmax=1, square=True)