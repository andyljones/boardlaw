import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rebar import dotdict, paths
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

def elos(run_name, target=None, filter='.*'):
    run_name = paths.resolve(run_name)
    games, wins = database.symmetric_pandas(run_name)
    games, wins = mask(games, wins, filter)

    soln = activelo.solve(games.values, wins.values)
    soln = pandas(soln, games.index)

    if isinstance(target, int):
        μ, σ = difference(soln, soln.μ.index[target])
    elif isinstance(target, str):
        μ, σ = difference(soln, target)
    else:
        μ, σ = soln.μ, np.diag(soln.Σ)**.5

    return pd.concat({'μ': μ, 'σ': σ}, 1)

def plot_elo_progress():
    import numpy as np
    import matplotlib.pyplot as plt
    with plt.style.context('seaborn-poster'):
        df = stats.dataframe(prefix='elo-mohex', rule='30s')['mean_std'].mul(400/np.log(10))
        fig, ax = plt.subplots()
        df['elo-mohex/μ'].plot(ax=ax)
        ax.fill_between(df.index, df['elo-mohex/μ-'], df['elo-mohex/μ+'], alpha=.2)
        ax.set_ylabel('agent elo v. perfect play')
        ax.set_title('Training progress on 5x5 Hex')
        ax.axhline(0, color='k', alpha=.5)

