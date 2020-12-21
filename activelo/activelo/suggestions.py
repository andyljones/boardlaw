from activelo.common import numpyify
import pandas as pd
import torch
import torch.distributions
import torch.testing
import numpy as np
import scipy as sp
import scipy.stats
from . import solvers, common
from rebar import arrdict

def safe_divide(x, y):
    r = np.zeros_like(x)
    np.divide(x, y, out=r, where=(y > 0))
    return r

def improvement(soln):
    if isinstance(soln.μd, pd.DataFrame):
        return common.pandify(improvement(common.numpyify(soln)), soln.μd.index)
    # This comes from looking at a rank-1 update to the information present in 
    # prior.
    e = np.exp(-soln.μd)
    likelihood_info = 1/(1/e + 2 + e)
    return soln.σd**2 * likelihood_info

def suggest(soln):
    if isinstance(soln.μ, pd.Series):
        row, col = suggest(common.numpyify(soln))
        return soln.μ.index[row], soln.μ.index[col]
    idx = improvement(soln).argmax()
    return np.unravel_index(idx, soln.μd.shape) 

def test_improvement():
    # Get more information from less-certain pairs
    first = arrdict.arrdict(μd=0., σd=1.)
    second = arrdict.arrdict(μd=0., σd=2.)
    assert improvement(first) < improvement(second)

    # Get more information from closer-in-rating pairs 
    first = arrdict.arrdict(μd=0., σd=1.)
    second = arrdict.arrdict(μd=1., σd=1.)
    assert improvement(first) > improvement(second)

    # Get the same information when flipping the difference 
    first = arrdict.arrdict(μd=-1., σd=1.)
    second = arrdict.arrdict(μd=+1., σd=1.)
    assert improvement(first) == improvement(second)