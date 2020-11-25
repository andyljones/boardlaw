"""
So what does a general-purpose evaluator look like?
* Got a collection of agents and a collection of games already played among them
* One is a 'reference strength' agent that has Elo 1000 (or whatever ranking system you choose to use)
* Goal is to estimate the Elo of a set of target agents to a specific level of confidence, as fast as possible
* This is basically the ResponseGraphUCB problem.
"""
import numpy as np
import scipy as sp
import scipy.stats
import networkx as nx
from rebar import arrdict

def random_problem(mean=1000, std=400, n_agents=5, n_games=20, concentration=.7):
    ranks = np.random.normal(mean, std, (n_agents,))

    concentration = .7
    alpha = np.full(n_agents*(n_agents-1)//2, concentration)
    ps = sp.stats.dirichlet(alpha).rvs(())

    games = sp.random.binomial(n_games, ps)

    k = 0
    wins = np.zeros_like(games)
    edges = np.zeros((len(games), 2), dtype=int)
    for i in range(n_agents):
        for j in range(i+1, n_agents):
            winrate = sp.special.expit((ranks[i] - ranks[j])/std)
            
            wins[k] = sp.random.binomial(games[k], winrate)
            edges[k] = (i, j)
            
            k += 1
            
    return arrdict.arrdict(
            ranks=ranks,
            wins=wins,
            games=games,
            edges=edges)

def solve(prob):
    pass