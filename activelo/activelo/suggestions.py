from activelo.common import numpyify
import pandas as pd
import torch
import torch.distributions
import torch.testing
import numpy as np
import scipy as sp
import scipy.stats
from . import solvers, common

def safe_divide(x, y):
    r = np.zeros_like(x)
    np.divide(x, y, out=r, where=(y > 0))
    return r

def sensitivities(Σ):
    # Residual v. the average agent
    Σ = Σ - np.outer(Σ.mean(0), Σ.mean(0))/Σ.mean()
    # Change in trace if we conditioned I-J out
    return ((Σ[:, None, :] - Σ[None, :, :])**2).sum(-1)

def improvement(soln):
    e = np.exp(-soln.μd)
    # This is the Fisher info for one game
    fisher_info = 1/(1/e + 2 + e)
    return fisher_info*sensitivities(soln.Σ)

def suggest(soln):
    if isinstance(soln.μ, pd.Series):
        row, col = suggest(common.numpyify(soln))
        return soln.μ.index[row], soln.μ.index[col]
    idx = improvement(soln).argmax()
    return np.unravel_index(idx, soln.μd.shape) 

def test_sensitivities():
    Σ = torch.tensor([[1, 1/2], [1/2, 2]])

    trace = sensitivities(Σ)

def test_improvement():
    k = 10
    n = torch.tensor([[0, k], [k, 0]])
    w = torch.tensor([[0, k//2], [k//2, 0]])

    solver = solvers.Solver(2)

    first = solver(n, w)
    second = solver(2*n, 2*n)

    actual = improvement(first)