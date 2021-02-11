import numpy as np
import activelo
from . import solvers

def stds():
    n, w = solvers.example('data/2021-02-06 12-16-42 wan-ticks.npz')

    σs = []
    contrast = 0
    for _ in range(20):
        soln = activelo.solve(n, w)
        μ, Σ = soln.μ, soln.Σ 
        σ2 = np.diag(Σ) + Σ[contrast, contrast] - 2*Σ[contrast]
        σs.append(σ2[-1]**.5)
    σs = np.array(σs)

    return σs