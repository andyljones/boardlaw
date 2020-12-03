"""
So what does a general-purpose evaluator look like?
* Got a collection of agents and a collection of games already played among them
* One is a 'reference strength' agent that has Elo 1000 (or whatever ranking system you choose to use)
* Goal is to estimate the Elo of a set of target agents to a specific level of confidence, as fast as possible
* This is basically the ResponseGraphUCB problem.
"""
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
import networkx as nx
from torch._C import Value
from rebar import arrdict
import torch
from torch import nn
from torch.distributions import Uniform, SigmoidTransform, AffineTransform, TransformedDistribution
from . import vb

def random_ranks(n_agents=10):
    deltas = torch.randn((n_agents,))/n_agents**.5
    totals = deltas.cumsum(0) 
    totals = totals - totals.min()
    return torch.sort(totals).values

def log_ranks(n_agents=10):
    return torch.linspace(1, 26, n_agents).float().log()

def winrate(black, white):
    return 1/(1 + np.exp(-(black - white)))

def simulate(black, white, n_games):
    return torch.distributions.Binomial(n_games, winrate(black, white)).sample()

def loss(wins, games, ranks, prior=1):
    rates = winrate(ranks[:, None], ranks[None, :])
    losses = -(wins + prior)*torch.log(rates) - (games - wins + prior)*torch.log(1 - rates)
    return losses.sum()

def infer(wins, games, ranks):
    ranks = nn.Parameter(ranks)
    optim = torch.optim.LBFGS([ranks])

    def closure():
        optim.zero_grad()
        l = loss(wins, games, ranks)
        l.backward()
        return l

    # Only need one step to converge; everything's done internally.
    optim.step(closure)

    return ranks.data

def unravel(index, shape):
    n_agents = shape[-1]
    black, white = index // n_agents, index % n_agents
    return black, white

def min_suggest(wins, games, ranks):
    return unravel(games.argmin(), games.shape)

def grad_suggest(wins, games, ranks):
    wins = nn.Parameter(wins)
    l = loss(wins, games, ranks)
    l.backward()

    sensitivities = wins.grad.abs()
    return unravel(sensitivities.argmax(), games.shape)

def solve(truth, games_per=256, σbar_tol=.1):
    n_agents = len(truth)
    games_per = 256
    n_agents = len(truth)
    wins = torch.zeros((n_agents, n_agents))
    games = torch.zeros((n_agents, n_agents))

    solns = []
    solver = vb.Solver(n_agents)
    ranks = torch.full((n_agents,), 0.)
    i = 1
    while True:
        soln = solver(games, wins)
        ranks = torch.as_tensor(soln.μ)

        black, white = vb.suggest(soln, games_per)
        black_wins = simulate(truth[black], truth[white], games_per)
        wins[black, white] += black_wins
        wins[white, black] += games_per - black_wins
        games[black, white] += games_per
        games[white, black] += games_per

        soln['n'] = games.clone()
        soln['w'] = wins.clone()
        soln['σbar'] = (soln.σd**2).mean(-1).mean(-1)**.5
        soln['resid_var'] = 1 - np.corrcoef(ranks, truth)[0, 1]**.2
        solns.append(arrdict.arrdict({k: v for k, v in soln.items() if k != 'trace'}))
        print(soln.σbar, soln.resid_var)
        if soln.σbar < σbar_tol:
            break
        if i > 1 and np.isnan(soln.resid_var):
            raise ValueError('Crashed')
        
        i += 1
        
    solns = arrdict.stack(solns)

    return solns

def fixed_benchmark():
    truth = log_ranks(10)
    solve(truth)

def random_benchmark():
    counts = []
    for _ in range(100):
        ranks = random_ranks(n_agents=10)
        counts.append(solve(ranks))
    q = np.quantile(counts, .95)
    return q

def example():
    from . import database

    run_name = '2020-11-27 21-32-59 az-test'
    winrate = database.symmetric_winrate(run_name).fillna(0).values
    games = database.symmetric_games(run_name).values
    wins = winrate*games

    advi, approx = pymc_solve(wins, games)

    trace = approx.sample(5000)
    
    pymc_plot(trace)
