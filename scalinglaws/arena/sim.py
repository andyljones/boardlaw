"""
So what does a general-purpose evaluator look like?
* Got a collection of agents and a collection of games already played among them
* One is a 'reference strength' agent that has Elo 1000 (or whatever ranking system you choose to use)
* Goal is to estimate the Elo of a set of target agents to a specific level of confidence, as fast as possible
* This is basically the ResponseGraphUCB problem.
"""
import pystan
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

def stan_model():
    ocode = """
    data {
        int<lower=0> N;
        int wins[N, N];
        int games[N, N];
    }
    parameters {
        real mu[N];
        real<lower=0> sigma[N];
        vector[N] skill;
    }
    model {
        vector[N] d;
        
        sigma ~ gamma(2, 1);
        skill ~ normal(mu, sigma);

        for (i in 1:N) {
            d = -(skill - skill[i]);
            wins[i] ~ binomial_logit(games[i], d);
        }
    }
    """
    return pystan.StanModel(model_code=ocode)

def unpack(result):
    if isinstance(result, dict):
        means = dict(zip(result['mean_par_names'], result['mean_pars']))
    else:
        s = result.summary()
        means = dict(zip(s['summary_rownames'], s['summary'][:-1, 0]))

    dicts = {}
    for k, v in means.items():
        name, pos = k[:-1].split('[')
        dicts.setdefault(name, {})[int(pos)] = v

    arrs = {}
    for name, d in dicts.items():
        arrs[name] = np.zeros(len(d))
        for i, v in d.items():
            arrs[name][i-1] = v

    return arrs

def stan_solve(wins, games):
    model = stan_model()

    data = dict(
        N=wins.size(0),
        wins=wins.int().numpy(),
        games=games.int().numpy())

    raw = model.vb(data=data, diagnostic_file='output/stan-diag.txt', tol_rel_obj=1e-5, verbose=True)
    result = unpack(raw)

    return result

def pymc_solve(wins, games):
    n_agents = wins.shape[0]
    with pm.Model() as model:
        sigma = pm.InverseGamma('sigma', alpha=1, beta=1, shape=(n_agents-1,))
        mu = pm.Normal('mu', mu=0, sigma=10, shape=(n_agents-1,))
        
        skills = pm.math.concatenate([
            pm.math.zeros_like(mu[:1]), 
            pm.Normal('skills', mu, sigma, shape=(n_agents-1,))])
        
        diffs = skills[:, None] - skills[None, :]
        rate = 1/(1 + 10**(-diffs))
        
        outcomes = pm.Binomial('outcomes', n=games, p=rate, observed=wins, shape=(n_agents, n_agents))
        
        mf = pm.fit(n=25000)

    return mf

def pymc_plot(mf):
    import pandas as pd
    import seaborn as sns
    trace = mf.sample(5000)

    df = (pd.DataFrame(trace['skills'])
            .rename_axis(index='sample', columns='agent')
            .stack()
            .reset_index()
            .rename(columns={0: 'elo'}))
    ax = sns.violinplot(data=df, x='agent', y='elo', inner=None, linewidth=1)
    ax.set_xticks([])

def benchmark():
    counts = []
    for _ in range(100):
        ranks = random_ranks(n_agents=10)
        counts.append(solve(ranks))
    q = np.quantile(counts, .95)
    return q

def example():
    from . import database

    winrate = torch.as_tensor(database.symmetric_winrate(-1).fillna(0).values)
    games = torch.as_tensor(database.symmetric_games(-1).values)
    wins = winrate*games

    ranks = torch.zeros((wins.shape[0],))
    ranks = infer(wins, games, ranks)

    result = stan_solve(wins, games)
    plt.plot((result['mu'] - result['mu'][0])/(result['mu'][-1] - result['mu'][0]))
    plt.plot((ranks - ranks[0])/(ranks[-1] - ranks[0]))
