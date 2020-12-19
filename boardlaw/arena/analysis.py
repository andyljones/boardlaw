import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rebar import dotdict, paths
from . import database
import activelo

def to_pandas(soln, games):
    return dotdict.dotdict(
        μ=pd.Series(soln.μ, games.index),
        Σ=pd.DataFrame(soln.Σ, games.index, games.index))

def condition(soln, name):
    μ, Σ = soln.μ, soln.Σ 
    σ2 = np.diag(Σ) + Σ.loc[name, name] - 2*Σ[name]
    μc = μ - μ[name]
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
    soln = to_pandas(soln, games)

    if isinstance(target, int):
        μ, σ = condition(soln, soln.μ.index[target])
    elif isinstance(target, str):
        μ, σ = condition(soln, target)
    else:
        μ, σ = soln.μ, np.diag(soln.Σ)**.5

    return pd.concat({'μ': μ, 'σ': σ}, 1)

