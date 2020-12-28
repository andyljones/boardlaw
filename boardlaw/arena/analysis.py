import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pavlov import runs, stats
from rebar import dotdict
from . import database
import activelo

def pandas(soln, names):
    return dotdict.dotdict(
        μ=pd.Series(soln.μ, names),
        Σ=pd.DataFrame(soln.Σ, names, names))

def difference(soln, contrast, name=None):
    μ, Σ = soln.μ, soln.Σ 
    σ2 = np.diag(Σ) + Σ.loc[contrast, contrast] - 2*Σ[contrast]
    μc = μ - μ[contrast]
    if name:
        return μc[name], σ2[name]**.5
    else:
        return μc, σ2**.5

def mask(games, wins, filter):
    mask = games.index.str.match(filter)
    games, wins = games.loc[mask, mask], wins.loc[mask, mask]
    return games, wins

def elos(run, target=None, filter='.*'):
    run = runs.resolve(run)
    games, wins = database.symmetric_pandas(run)
    games, wins = mask(games, wins, filter)

    soln = activelo.solve(games.values, wins.values)
    soln = pandas(soln, games.index)

    if isinstance(target, int):
        μ, σ = difference(soln, soln.μ.index[target])
    elif isinstance(target, str):
        μ, σ = difference(soln, target)
    else:
        μ, σ = soln.μ, pd.Series(np.diag(soln.Σ)**.5, games.index)

    return pd.concat({'μ': μ, 'σ': σ}, 1)

def plot_elo_progress(run_names=[-1]):
    with plt.style.context('seaborn-poster'):
        fig, ax = plt.subplots()
        ax.set_ylabel('agent elo v. perfect play')
        ax.set_title('Training progress on 7x7 Hex')
        ax.axhline(0, color='k', alpha=.5)
        ax.grid(axis='y')

        for run_name in run_names:
            df = stats.dataframe(run_name, prefix='elo-mohex')['mean_std'].ffill()
            hours = df.index.total_seconds()/3600
            #df['elo-mohex/μ'].plot(ax=ax, label=run_name)
            ax.fill_between(hours, df['elo-mohex/μ-'], df['elo-mohex/μ+'], alpha=.2)
            ax.plot(hours, df['elo-mohex/μ'], label=runs.resolve(run_name))
            ax.set_xlim(0, hours.max())
            ax.set_xlabel('hours')
        
        ax.legend()