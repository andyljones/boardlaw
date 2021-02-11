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
    games, wins = database.symmetric(run)
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
