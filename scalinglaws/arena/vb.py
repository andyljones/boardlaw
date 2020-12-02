import numpy as np
import sympy as sym

μ0 = 0
σ0 = 1

def test_d_integral():
    import numpy as np
    import scipy as sp
    import scipy.stats

    Λ = np.array([[1, 0], [0, 1]])
    μ = np.array([0, 0])

    def ϕ(d):
        return 1/(1 + np.exp(-d))

    def integrand(s):
        return np.log(ϕ(s[..., 0] - s[..., 1]))
        

    N = sp.stats.multivariate_normal(μ, np.linalg.inv(Λ))
    actual = integrand(N.rvs(1000)).mean()

    mult = np.sqrt(32*np.pi/(Λ[0, 0] + 2*Λ[0, 1] + Λ[1, 1]))

    λd = (Λ[0, 0]*Λ[1, 1] - Λ[0, 1]**2)/(Λ[0, 0] + Λ[1, 1] + 2*Λ[0, 1])
    μd = μ[0] - μ[1]

    Nd = sp.stats.norm(μd, 1/λd**.5)
    expected = mult*integrand(Nd.rvs(1000)).mean()

def winrate(μ, Σ):
    # μ = MatrixSymbol('μ', 2, 1)
    # Λ = MatrixSymbol('Λ', 2, 2)
    # d, m = symbols('d m')
    # I = Matrix([[1/2], [1/2]])
    # D = Matrix([[1/2], [-1/2]])

    # integrand = trace(Matrix(-1/2*(m*I + d*D - μ).T @ Λ @ (m*I + d*D - μ)))
    # integrand = integrand.subs(Λ[0, 1], Λ[1, 0])
    # integrand = poly(integrand, m).coeffs()

    # a, b, c = symbols('a b c')
    # gaussian_integrand = poly(-a*(m - b)**2 + c, m).coeffs()

    # [(a, b, c)] = solve([Eq(l, r) for l, r in zip(gaussian_integrand, integrand)], (a, b, c))

    # a = simplify(a)
    # c = simplify(factor(c), rational=True)
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
    # 1/(2*σ0)*E(-(s - μ0)**2)
    prior = -1/(2*σ0)*((μ - μ0)**2 + Σ**2)

    return likelihood.sum() + prior.sum()
