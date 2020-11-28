"""
So what does a general-purpose evaluator look like?
* Got a collection of agents and a collection of games already played among them
* One is a 'reference strength' agent that has Elo 1000 (or whatever ranking system you choose to use)
* Goal is to estimate the Elo of a set of target agents to a specific level of confidence, as fast as possible
* This is basically the ResponseGraphUCB problem.
"""
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

def winrate(black, white):
    return 1/(1 + 10**(-(black - white)/400))

def simulate(black, white, n_games):
    return torch.distributions.Binomial(n_games, winrate(black, white)).sample()

def infer(wins, games, initial, prior=1):
    ranks = nn.Parameter(initial)
    optim = torch.optim.LBFGS([ranks])

    def closure():
        rates = winrate(ranks[:, None], ranks[None, :])
        losses = -(wins + prior)*torch.log(rates) - (games - wins + prior)*torch.log(1 - rates)
        loss = losses.sum()
        optim.zero_grad()
        loss.backward()
        return loss

    optim.step(closure)

    return ranks

def solve(ranks, games_per=256):
    n_agents = len(ranks)
    wins = torch.zeros((n_agents, n_agents))
    games = torch.zeros((n_agents, n_agents))

    estimates = torch.full((n_agents,), 1000.)
    i = 1
    while True:
        estimates = infer(wins, games, estimates)

        black, white = torch.randint(n_agents, (2,))
        black_wins = simulate(ranks[black], ranks[white], games_per)
        wins[black, white] += black_wins
        wins[white, black] += games_per - black_wins
        games[black, white] += games_per
        games[white, black] += games_per

        err = (estimates - estimates[0]) - (ranks - ranks[0])
        ratio = err.pow(2).mean()/(ranks - ranks[0]).pow(2).mean()
        if ratio < .01:
            break

        i += 1

    return i

def benchmark():
    counts = []
    for _ in range(100):
        ranks = random_ranks(n_agents=10)
        counts.append(solve(ranks))
    q = np.quantile(counts, .95)
    return q

