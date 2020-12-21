import torch
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import activelo

def generated_example():
    N = 20
    truth = torch.randn(N)
    n = torch.randint(1, 50, (N, N))

    d = truth[:, None] - truth[None, :]
    w = torch.distributions.Binomial(n, 1/(1 + np.exp(-d))).sample()

    trace = activelo.solve(n, w)

    activelo.plot(trace)

def saved_example(filename):
    raw = np.load(filename)
    n = torch.as_tensor(raw['n'])
    w = torch.as_tensor(raw['w'])
    torch.set_rng_state(torch.as_tensor(raw['rng']))

    soln = activelo.solve(n, w)

    activelo.plot(soln)

    return soln

def saved_examples():
    # Generated during development of my AlphaZero agent
    saved_example('data/2020-12-07 21-59-18 az-test symmetric.npz')

    # A 100-agent problem that seems prone to either line search failures or negdef Σ.
    saved_example('data/line-search-failure.npz')


