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

def posterior_var_ratio(μd, σd, G):
    ϕ = lambda d: 1/(1 + np.exp(-d))
    diϕ = lambda r: 1/r + 1/(1 - r)
    dϕ = lambda d: -np.exp(-d)/(1 + np.exp(-d))**2

    num = solvers.σ0**2 * dϕ(μd)**2 * σd**2 
    denom = solvers.σ0**2 + dϕ(μd)**2 *σd**2 * G

    posterior_var = diϕ(ϕ(μd))**2 * num/denom

    return safe_divide(posterior_var, σd**2)

def improvement(soln, G):
    if isinstance(soln.μ, pd.Series):
        return common.pandify(improvement(common.numpyify(soln), G), soln.μ.index)
    return safe_divide(sensitivities(soln.Σ), posterior_var_ratio(soln.μd, soln.σd, G))

def suggest(soln, G):
    if isinstance(soln.μ, pd.Series):
        row, col = suggest(common.numpyify(soln), G)
        return soln.μ.index[row], soln.μ.index[col]
    idx = improvement(soln, G).argmax()
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

    expected = second.σd[0, 1]/first.σd[0, 1]
    actual = improvement(first, k)

    torch.testing.assert_allclose(expected, actual)