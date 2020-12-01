import numpy as np
import sympy as sym

μ0 = 0
σ0 = 1

def winrate(μ, Σ):
    pass

def lossrate(μ, Σ):
    pass

def joint_prob(n, w, μ, Σ):
    likelihood = w*winrate(μ, Σ) + (n - w)*lossrate(μ, Σ)

    # from sympy.stats import E, Normal
    # s, μ, μ0, σ, σ0 = symbols('s μ μ_0 σ σ_0')
    # s = Normal('s', μ, σ)
    # 1/(2*σ0)*E((s - μ0)**2)
    prior = 1/(2*σ0)*((μ - μ0)**2 + Σ**2)
