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

def generate_problem(mean=1000, std=400, n_agents=5, n_games=20):
    ranks = np.random.normal(mean, std, (n_agents,))
    
    #TODO: This won't reliably be connected :/
    alpha = np.full(n_agents*(n_agents-1), .1)
    sp.stats.dirichlet(alpha).rvs().round(3)

    return ranks

