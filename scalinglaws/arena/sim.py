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
from rebar import arrdict
import torch
from torch import nn
from torch.distributions import Uniform, SigmoidTransform, AffineTransform, TransformedDistribution

def random_ranks(std=400, n_agents=10):
    deltas = std/2*torch.randn((n_agents,))
    totals = deltas.cumsum(0) 
    totals = totals - totals.min()
    return torch.sort(totals).values

def log_ranks(n_agents=10):
    return 200*torch.linspace(1, 26, n_agents).float().log()

def winrate(black, white):
    return 1/(1 + 10**(-(black - white)/400))

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

def solve(truth, games_per=256):
    n_agents = len(truth)
    wins = torch.zeros((n_agents, n_agents))
    games = torch.zeros((n_agents, n_agents))

    ranks = torch.full((n_agents,), 1000.)
    i = 1
    while True:
        ranks = infer(wins, games, ranks)

        black, white = min_suggest(wins, games, ranks)
        black_wins = simulate(truth[black], truth[white], games_per)
        wins[black, white] += black_wins
        wins[white, black] += games_per - black_wins
        games[black, white] += games_per
        games[white, black] += games_per

        err = (ranks - ranks[0]) - (truth - truth[0])
        ratio = err.pow(2).mean()/(truth - truth[0]).pow(2).mean()
        if ratio < .01:
            break

        i += 1

    return i

def pymc_solve(wins, games):
    n_agents = wins.shape[0]
    with pm.Model() as model:
        skills = pm.math.concatenate([
            np.zeros((1,)), 
            pm.Normal('skills', 0, 10, shape=(n_agents-1,))])
        
        diffs = skills[:, None] - skills[None, :]
        rate = pm.math.invlogit(diffs) # need to multiply by 400*log(10) to get Elos
        
        pm.Binomial('outcomes', n=games, p=rate, observed=wins, shape=(n_agents, n_agents))

        advi = pm.FullRankADVI()
        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,
            std=advi.approx.std.eval)
        conv = pm.callbacks.CheckParametersConvergence(diff='absolute', tolerance=.01, every=1000)

        approx = advi.fit(50000, callbacks=[tracker, conv])


    return advi, approx

def pymc_plot_means(trace):
    import pandas as pd
    import seaborn as sns

    df = (pd.DataFrame(trace['skills'])
            .rename_axis(index='sample', columns='agent')
            .stack()
            .reset_index()
            .rename(columns={0: 'elo'}))
    ax = sns.violinplot(data=df, x='agent', y='elo', inner=None, linewidth=1)
    ax.set_xticks([])

def pymc_diff_stds(advi):
    mu = advi.approx.mean.eval()
    cov = advi.approx.cov.eval()

    mu_d = mu[:, None] - mu[None, :]
    std_d = np.diag(cov)[:, None] + np.diag(cov)[None, :] - 2*cov

    return std_d

def benchmark():
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
