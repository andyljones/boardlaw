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
    if isinstance(soln.μ, pd.Series):
        return common.pandify(improvement(common.numpyify(soln)), soln.μ.index)
    e = np.exp(-soln.μd)
    #TODO: This is not in any way correct. It's the Fisher info v. the d's, when we're
    # really after the Fisher info v. the x ~ N(μ, Σ). Go re-read 'Asymptotic Analysis 
    # of Objectives based on Fisher Information'
    fisher_info = 1/(1/e + 2 + e)
    # And I definitely shouldn't be multiplying the info by the var like this
    # I think the proper way is gonna involve adding various rank-one updates to the
    # Σ^-1 and seeing which has the best trace/det/whatever. 
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