import numpy as np
import sympy as sym

μ0 = 0
σ0 = 1

def winrate(μ, Σ):
    # from sympy import *
    # from sympy import stats
    # from math import inf
    #
    # μ = MatrixSymbol('μ', 2, 1)
    # Λ = MatrixSymbol('Λ', 2, 2)
    # d, m = symbols('d m')
    # I = Matrix([[1/2], [1/2]])
    # D = Matrix([[1/2], [-1/2]])
    #
    # integrand = trace(Matrix(-1/2*(m*I + d*D - μ).T @ Λ @ (m*I + d*D - μ)))
    # integrand = poly(integrand, m)
    #
    # a, b, c = symbols('a b c')
    # pattern = poly(-a*(m - b)**2 + c, m)
    #
    # [(a, b, c)] = solve([Eq(l, r) for l, r in zip(pattern.coeffs(), integrand.coeffs())], (a, b, c))
    #
    # c = simplify(factor(c))
    # integral = E**c * sqrt(pi/a)

    pass

def lossrate(μ, Σ):
    pass

def joint_prob(n, w, μ, Σ):
    likelihood = w*winrate(μ, Σ) + (n - w)*lossrate(μ, Σ)

    # Proof:
    # from sympy.stats import E, Normal
    # s, μ, μ0, σ, σ0 = symbols('s μ μ_0 σ σ_0')
    # s = Normal('s', μ, σ)
    # 1/(2*σ0)*E((s - μ0)**2)
    prior = 1/(2*σ0)*((μ - μ0)**2 + Σ**2)
