from rebar import arrdict, dotdict
import numpy as np
import sympy as sym
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Function

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
        self.j, self.k = torch.as_tensor(np.indices((N, N)).reshape(2, -1))

    def forward(self, μ, Σ):
        j, k = self.j, self.k
        μd = μ[j] - μ[k]
        σd = Σ[j, j] - Σ[j, k] - Σ[k, j] + Σ[k, k]
        return μd, σd.clamp(1e-6, None)**.5

    def as_square(self, x):
        return x.reshape(self.N, self.N)

def evaluate(interp, μd, σd):
    return torch.as_tensor(interp(μd.detach().numpy(), σd.detach().numpy(), grid=False)).float()

class GaussianExpectation(Function):

    @staticmethod
    def auxinfo(f, K=101, S=1000):
        μ = np.linspace(*μ_lims, K)
        σ = np.logspace(*σ_lims, K, base=10)

        #TODO: Importance sample these zs
        zs = np.linspace(-5, +5, S)
        pdf = sp.stats.norm.pdf(zs)[None, None, :]
        ds = (μ[:, None, None] + zs[None, None, :]*σ[None, :, None])
        fs = (f(ds)*pdf/pdf.sum()).sum(-1)
        f = sp.interpolate.RectBivariateSpline(μ, σ, fs, kx=1, ky=1)

        dμs = (fs[2:, :] - fs[:-2, :])/(μ[2:] - μ[:-2])[:, None]
        dμ = sp.interpolate.RectBivariateSpline(μ[1:-1], σ, dμs, kx=1, ky=1)
        dσs = (fs[:, 2:] - fs[:, :-2])/(σ[2:] - σ[:-2])[None, :]
        dσ = sp.interpolate.RectBivariateSpline(μ, σ[1:-1], dσs, kx=1, ky=1)

        return dotdict.dotdict(
            μ=μ, σ=σ, 
            f=f, dμ=dμ, dσ=dσ, 
            fs=fs, dμs=dμs, dσs=dσs)

    @staticmethod
    def forward(ctx, μd, σd, aux):
        ctx.save_for_backward(μd, σd)
        ctx.aux = aux
        return evaluate(aux.f, μd, σd)

    @staticmethod
    def backward(ctx, dldf):
        μd, σd = ctx.saved_tensors
        dfdμ = evaluate(ctx.aux.dμ, μd, σd)
        dfdσ = evaluate(ctx.aux.dσ, μd, σd)

        dldμ = dldf*dfdμ
        dldσ = dldf*dfdσ

        return dldμ, dldσ, None

    def plot(self):
        Y, X = np.meshgrid(self.μ, self.σ)
        (t, b), (l, r) = μ_lims, σ_lims
        plt.imshow(np.exp(self.fs), extent=(l, r, b, t), vmin=0, vmax=1, cmap='RdBu', aspect='auto')
        plt.colorbar()

gaussian_expectation = GaussianExpectation.apply

def expected_log_likelihood(n, w, μ, Σ):
    self = expected_log_likelihood
    N = n.shape[0]
    if not hasattr(self, '_differ') or self._differ.N != N:
        self._differ = Differ(n.shape[0])
        self._aux = GaussianExpectation.auxinfo(lambda d: -np.log(1 + np.exp(-d)))
    differ, aux = self._differ, self._aux

    μd, σd = differ(μ, Σ)
    wins = w*differ.as_square(gaussian_expectation(μd, σd, aux))
    losses = (n - w)*differ.as_square(gaussian_expectation(-μd, σd, aux))
    return wins + losses

def cross_entropy(n, w, μ, Σ):

    # Proof:
    # from sympy.stats import E, Normal
    # s, μ, μ0, σ, σ0 = symbols('s μ μ_0 σ σ_0')
    # s = Normal('s', μ, σ)
    # 1/(2*σ0)*E(-(s - μ0)**2)
    expected_prior = -1/(2*σ0)*((μ - μ0)**2 + Σ**2)

    return -expected_log_likelihood(n, w, μ, Σ).sum() - expected_prior.sum()

def entropy(Σ):
    return 1/2*torch.logdet(2*np.pi*np.e*Σ)

def elbo(n, w, μ, Σ):
    return -cross_entropy(n, w, μ, Σ) + entropy(Σ)

@torch.no_grad()
def project(Σ):
    symmetric = (Σ + Σ.T)/2
    λ, v = torch.symeig(symmetric, True)
    return v @ torch.diag(λ.clamp(1e-3, None)) @ v.T

def solve(n, w, tol=1e-3):
    N = n.shape[0]

    μ = torch.nn.Parameter(torch.zeros((N,)))
    Σ = torch.nn.Parameter(torch.eye(N))

    optim = torch.optim.Adam([μ, Σ], .01)

    ls = []
    for i in range(200):
        l = -elbo(n, w, μ, Σ)
        optim.zero_grad()
        l.backward()
        optim.step()
        Σ = project(Σ)

        ls = (ls + [l.detach()])[-10:]
        if len(ls) > 1 and abs(ls[-1] - ls[0]) < tol*ls[0]:
            break
        if i % 10 == 0:
            print(l)
    else:
        print('Didn\'t converge')
    
    return dotdict.dotdict(μ=μ.detach(), Σ=Σ.detach())


def test():
    N = 5

    s = np.random.randn(N)

    n = np.random.randint(1, 10, (N, N))

    d = s[:, None] - s[None, :]
    w = sp.stats.binom(n, 1/(1 + np.exp(-d))).rvs()

    soln = solve(n, w)

    differ = Differ(N)
    μd, σd = differ.as_square(*differ(soln.μ, soln.Σ))

    plt.plot(soln.μ)
    plt.imshow(soln.σd)
