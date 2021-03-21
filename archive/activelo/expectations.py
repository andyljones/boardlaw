import matplotlib.pyplot as plt
import torch
import torch.testing
from torch.autograd import Function
import numpy as np
import scipy as sp
import scipy.interpolate
from rebar import dotdict
from functools import wraps

μ_lims = [-25, +25]
σ2_lims = [-4, +2]

def evaluate(interp, μd, σd):
    return torch.as_tensor(interp(μd.detach().numpy(), σd.detach().numpy(), grid=False)).double()

class Normal(Function):

    @staticmethod
    def auxinfo(f, K=1001, S=50):
        μ = np.linspace(*μ_lims, K)
        σ2 = np.logspace(*σ2_lims, K, base=10)

        zs, ws = np.polynomial.hermite_e.hermegauss(S)
        ds = (μ[:, None, None] + zs[None, None, :]*σ2[None, :, None]**.5)

        scale = 1/(2*np.pi)**.5
        fs = scale*(f(ds)*ws).sum(-1)
        f = sp.interpolate.RectBivariateSpline(μ, σ2, fs, kx=1, ky=1)

        dμs = (fs[2:, :] - fs[:-2, :])/(μ[2:] - μ[:-2])[:, None]
        dμ = sp.interpolate.RectBivariateSpline(μ[1:-1], σ2, dμs, kx=2, ky=2)
        dσ2s = (fs[:, 2:] - fs[:, :-2])/(σ2[2:] - σ2[:-2])[None, :]
        dσ2 = sp.interpolate.RectBivariateSpline(μ, σ2[1:-1], dσ2s, kx=2, ky=2)

        return dotdict.dotdict(
            μ=μ, σ2=σ2, 
            f=f, dμ=dμ, dσ2=dσ2, 
            fs=fs, dμs=dμs, dσs=dσ2s)

    @staticmethod
    def forward(ctx, aux, μd, σ2d):
        ctx.save_for_backward(μd, σ2d)
        ctx.aux = aux
        return evaluate(aux.f, μd, σ2d)

    @staticmethod
    def backward(ctx, dldf):
        μd, σ2d = ctx.saved_tensors
        dfdμ = evaluate(ctx.aux.dμ, μd, σ2d)
        dfdσ2d = evaluate(ctx.aux.dσ2, μd, σ2d)

        dldμ = dldf*dfdμ
        dldσ2d = dldf*dfdσ2d

        return None, dldμ, dldσ2d

    @staticmethod
    def plot(aux):
        Y, X = np.meshgrid(aux.μ, aux.σ2)
        (t, b), (l, r) = μ_lims, σ2_lims
        plt.imshow(np.exp(aux.fs), extent=(l, r, b, t), vmin=0, vmax=1, cmap='RdBu', aspect='auto')
        plt.colorbar()

@wraps(Normal.auxinfo)
def normal(*args, **kwargs):
    aux = Normal.auxinfo(*args, **kwargs)
    
    def f(μd, σ2d):
        return Normal.apply(aux, μd, σ2d)

    return f

def test_expectation():
    # Test E[X]
    μ = torch.tensor(1.)
    σ2 = torch.tensor(2.)

    expected = μ

    expectation = normal(lambda x: x)
    actual = expectation(μ, σ2)

    torch.testing.assert_allclose(expected, actual)

def test_variance():
    # Test E[X**2]
    μ = torch.tensor(1.)
    σ2 = torch.tensor(2.)

    expected = μ**2 + σ2

    expectation = normal(lambda x: x**2)
    actual = expectation(μ, σ2)

    torch.testing.assert_allclose(expected, actual)