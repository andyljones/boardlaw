import numpy as np
import sympy as sym
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import torch
from torch import nn

μ0 = 0
σ0 = 2

μ_lims = [-5*σ0, +5*σ0]
σ_lims = [-2, +1]

def test_d_integral():
    Σ = np.array([[.6, .5], [.5, 1]])
    μ = np.array([0, 1])

    def ϕ(d):
        return 1/(1 + np.exp(-d))

    N = sp.stats.multivariate_normal(μ, Σ)
    s = N.rvs(10000)
    actual = np.log(ϕ(s[..., 0] - s[..., 1])).mean()

    R = np.array([[+1, -1]])
    σ2d = R @ Σ @ R.T
    μd = R @ μ

    Nd = sp.stats.norm(μd, σ2d**.5)
    d = Nd.rvs(10000)
    expected = np.log(ϕ(d)).mean()

    return expected, actual

class Differ(nn.Module):

    def __init__(self, N):
        super().__init__()

        self.N = N
        j, k = np.indices((N, N)).reshape(2, -1)
        row = np.arange(len(j))
        R = np.zeros((len(row), N))
        R[row, j] = 1
        R[row, k] = -1
        self.R = torch.as_tensor(R).float()

    def forward(self, μ, Σ):
        μd = self.R @ μ
        σd = torch.diag(self.R @ Σ @ self.R.T)
        return μd, σd

class GaussianExpectation(nn.Module):

    def __init__(self, f, K=101, S=1000):
        super().__init__()
        self.μ = np.linspace(*μ_lims, K)
        self.σ = np.logspace(*σ_lims, K, base=10)

        #TODO: Importance sample these zs
        zs = np.linspace(-5, +5, S)
        pdf = sp.stats.norm.pdf(zs)[None, None, :]
        ds = (self.μ[:, None, None] + zs[None, None, :]*self.σ[None, :, None])
        self.fs = (f(ds)*pdf/pdf.sum()).sum(-1)
        self._f = sp.interpolate.RectBivariateSpline(self.μ, self.σ, self.fs, kx=1, ky=1)

        self.dμs = (self.fs[2:, :] - self.fs[:-2, :])/(self.μ[2:] - self.μ[:-2])[:, None]
        self._dμ = sp.interpolate.RectBivariateSpline(self.μ[1:-1], self.σ, self.dμs, kx=1, ky=1)
        self.dσs = (self.fs[:, 2:] - self.fs[:, :-2])/(self.σ[2:] - self.σ[:-2])[None, :]
        self._dσ = sp.interpolate.RectBivariateSpline(self.μ, self.σ[1:-1], self.dσs, kx=1, ky=1)

    def _eval(self, interp, μd, σd):
        return torch.as_tensor(interp(μd.numpy(), σd.numpy(), grid=False)).float()

    def forward(self, μd, σd):
        self.save_for_backward(μd, σd)
        return self._eval(self._f, μd.numpy(), σd.numpy())

    def backward(self, dldf):
        μd, σd = self.saved_tensors
        dfdμ = self._eval(self._dμ, μd, σd)
        dfdσ = self._eval(self._dσ, μd, σd)

        dldμ = dldf*dfdμ
        dldσ = dldf*dfdσ

        return dldμ, dldσ

    def plot(self):
        Y, X = np.meshgrid(self.μ, self.σ)
        (t, b), (l, r) = μ_lims, σ_lims
        plt.imshow(np.exp(self.fs), extent=(l, r, b, t), vmin=0, vmax=1, cmap='RdBu', aspect='auto')
        plt.colorbar()

def expected_log_likelihood(n, w, μ, Σ):
    self = expected_log_likelihood
    if not hasattr(self, '_differ') or self._differ.N != n.shape[0]:
        self._differ = Differ(n.shape[0])
        self._expectation = GaussianExpectation(lambda d: -np.log(1 + np.exp(-d)))
    differ, expectation = self._differ, self._expectation

    μd, σd = differ(μ, Σ)
    return w*expectation(μd, σd) + (n - w)*expectation(-μd, σd)

def cross_entropy(n, w, μ, Σ):

    # Proof:
    # from sympy.stats import E, Normal
    # s, μ, μ0, σ, σ0 = symbols('s μ μ_0 σ σ_0')
    # s = Normal('s', μ, σ)
    # 1/(2*σ0)*E(-(s - μ0)**2)
    expected_prior = -1/(2*σ0)*((μ - μ0)**2 + Σ**2)

    return -expected_log_likelihood(n, w, μ, Σ).sum() - expected_prior.sum()

def entropy(Σ):
    _, logdet = np.linalg.slogdet(2*np.pi*np.e*Σ)
    return 1/2*logdet

def elbo(n, w, μ, Σ):
    return -cross_entropy(n, w, μ, Σ) + entropy(Σ)

def solve(n, w):
    N = n.shape[0]

    def pack(μ, Σ):
        return np.concatenate([μ, Σ.flatten()])

    def unpack(x):
        μ = x[:N]
        Σ = x[N:].reshape(N, N)
        Σsym = (Σ + Σ.T)/2

        λ, v = np.linalg.eigh(Σsym)
        Σproj = v @ np.diag(λ.clip(1e-3, None)) @ v.T 
        
        return μ, Σproj

    def f(x):
        μ, Σ = unpack(x)
        return -elbo(n, w, μ, Σ)

    μ0 = np.zeros((N,))
    Σ0 = np.eye(N)
    x0 = pack(μ0, Σ0)

    soln = sp.optimize.minimize(f, x0)
    μf, Σf = unpack(soln.x)

    return μf, Σf

def test():
    N = 5

    s = np.random.randn(N)

    n = np.random.randint(1, 10, (N, N))

    d = s[:, None] - s[None, :]
    w = sp.stats.binom(n, 1/(1 + np.exp(-d))).rvs()

    μf, Σf = solve(n, w)
