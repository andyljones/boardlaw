import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

def f(x):
    return -np.log(1 + np.exp(-x))

def p(x):
    return sp.stats.norm(0, 1).pdf(x)

def benchmark(μ, σ):
    return sp.integrate.quad(lambda x: f(x*σ + μ)*p(x), -10, +10)

def current(μ, σ, d=1000):
    zs = np.linspace(-10, +10, d)
    pdf = p(zs)
    fs = (f(zs*σ + μ)*pdf/pdf.sum()).sum(-1)
    return fs

def ideal(μ, σ, degree=10):
    x, w = np.polynomial.hermite_e.hermegauss(degree)
    return (f(x*σ + μ)*w).sum()/(2*np.pi)**.5

def plot_error(μ, σ):
    ideal_approx = [ideal(μ, σ, d) for d in range(1, 100)]
    current_approx = [current(μ, σ, d) for d in range(1, 100)]
    target = ideal(μ, σ, 100)

    plt.plot(abs(np.array(ideal_approx) - target), label='ideal')
    plt.plot(abs(np.array(current_approx) - target), label='current')
    plt.yscale('log')
    plt.legend()