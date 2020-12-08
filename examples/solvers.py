import torch
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import activelo

def plot(soln):
    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches(20, 5)

    ax = axes[0]
    ax.plot(soln.trace.l)
    ax.set_xlim(0, len(soln.trace.l)-1)
    ax.set_title('loss')

    ax = axes[1]
    ax.plot(soln.trace.relnorm)
    ax.set_xlim(0, len(soln.trace.relnorm)-1)
    ax.set_yscale('log')
    ax.set_title('norms')

    ax = axes[2]
    ax.errorbar(
        np.arange(soln.μd.shape[0]), 
        soln.μd[:, 0], yerr=soln.σd[0, :], marker='.', linestyle='')
    ax.set_xlim(0, len(soln.μ)-1)
    ax.set_title('μ')

    ax = axes[3]
    ax.imshow(soln.σd)
    ax.set_title('σd')

def generated_example():
    N = 20
    truth = torch.randn(N)
    n = torch.randint(1, 50, (N, N))

    d = truth[:, None] - truth[None, :]
    w = torch.distributions.Binomial(n, 1/(1 + np.exp(-d))).sample()

    trace = activelo.solve(n, w)

    plot(trace)

def saved_example(filename):
    raw = np.load(filename)
    n = torch.as_tensor(raw['n'])
    w = torch.as_tensor(raw['w'])
    torch.set_rng_state(torch.as_tensor(raw['rng']))

    soln = activelo.solve(n, w)

    plot(soln)

    return soln

def saved_examples():
    # Generated during development of my AlphaZero agent
    saved_example('data/2020-11-27 21-32-59 az-test symmetric.npz')

    # A 100-agent problem that seems prone to either line search failures or negdef Σ.
    saved_example('data/line-search-failure.npz')


