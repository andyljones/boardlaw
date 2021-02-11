import torch
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import resource_filename
import activelo

def generated_example():
    N = 20
    truth = torch.randn(N)
    n = torch.randint(1, 50, (N, N))

    d = truth[:, None] - truth[None, :]
    w = torch.distributions.Binomial(n, 1/(1 + np.exp(-d))).sample()

    trace = activelo.solve(n, w)

    activelo.plot(trace)

def example(filename):
    raw = np.load(resource_filename(__package__, filename))
    n = torch.as_tensor(raw['n'])
    w = torch.as_tensor(raw['w'])
    torch.set_rng_state(torch.as_tensor(raw['rng']))
    return n, w

def saved_example(filename):
    n, w = example(filename)

    soln = activelo.solve(n, w)

    activelo.plot(soln)

    return soln

def saved_examples():
    # Generated during development of my AlphaZero agent
    saved_example('data/2020-12-07 21-59-18 az-test symmetric.npz')

    # A 100-agent problem that seems prone to either line search failures or negdef Σ.
    saved_example('data/line-search-failure.npz')

def reuse_example():
    n, w = example('data/2021-02-06 12-16-42 wan-ticks.npz')

    σs = []
    contrast = 0
    soln = activelo.solve(n, w)
    for _ in range(20):
        # Strip out this `soln=soln` to suppress soln reuse
        soln = activelo.solve(n, w, soln=soln)
        μ, Σ = soln.μ, soln.Σ 
        σ2 = np.diag(Σ) + Σ[contrast, contrast] - 2*Σ[contrast]
        σs.append(σ2[-1]**.5)
    σs = np.array(σs)