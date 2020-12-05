from functools import wraps
from rebar import arrdict, dotdict
import numpy as np
import sympy as sym
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Function
from tqdm.auto import tqdm
import geotorch

μ0 = 0
σ0 = 1

μ_lims = [-5*σ0, +5*σ0]
σ2_lims = [-4, +2]

class Differ(nn.Module):

    def __init__(self, N):
        super().__init__()

        self.N = N
        j, k = torch.as_tensor(np.indices((N, N)).reshape(2, -1))
        self.j, self.k = j[j != k], k[j != k]

    def forward(self, μ, Σ):
        j, k = self.j, self.k
        μd = μ[j] - μ[k]
        σ2d = Σ[j, j] - Σ[j, k] - Σ[k, j] + Σ[k, k]
        return μd, σ2d

    def as_square(self, x, fill=0.):
        y = torch.full((self.N, self.N), fill).float()
        y[self.j, self.k] = x
        return y

def evaluate(interp, μd, σd):
    return torch.as_tensor(interp(μd.detach().numpy(), σd.detach().numpy(), grid=False)).float()

class NormalExpectation(Function):

    @staticmethod
    def auxinfo(f, K=1001, S=50):
        μ = np.linspace(*μ_lims, K)
        σ2 = np.logspace(*σ2_lims, K, base=10)

        zs, ws = np.polynomial.hermite_e.hermegauss(S)
        ds = (μ[:, None, None] + zs[None, None, :]*σ2[None, :, None]**.5)
        # Pick up a σ from the change of variables 
        scale = σ2[None, :]**.5/(2*np.pi)**.5
        fs = scale*(f(ds)*ws).sum(-1)
        f = sp.interpolate.RectBivariateSpline(μ, σ2, fs, kx=1, ky=1)

        dμs = (fs[2:, :] - fs[:-2, :])/(μ[2:] - μ[:-2])[:, None]
        dμ = sp.interpolate.RectBivariateSpline(μ[1:-1], σ2, dμs, kx=1, ky=1)
        dσ2s = (fs[:, 2:] - fs[:, :-2])/(σ2[2:] - σ2[:-2])[None, :]
        dσ2 = sp.interpolate.RectBivariateSpline(μ, σ2[1:-1], dσ2s, kx=1, ky=1)

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

@wraps(NormalExpectation.auxinfo)
def normal_expectation(*args, **kwargs):
    aux = NormalExpectation.auxinfo(*args, **kwargs)
    
    @wraps(NormalExpectation.apply)
    def f(*args):
        return NormalExpectation.apply(aux, *args)

    return f

class ELBO(nn.Module):
    
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.register_parameter('μ', nn.Parameter(torch.zeros((N,)).float()))
        self.register_parameter('Σ', nn.Parameter(torch.eye(N).float()))
        geotorch.positive_definite(self, 'Σ')

        self.differ = Differ(N)
        self.expectation = normal_expectation(lambda d: -np.log(1 + np.exp(-d)))

    def expected_log_likelihood(self, n, w):
        μd, σ2d = self.differ(self.μ, self.Σ)

        p = self.expectation(μd, σ2d)
        q = self.expectation(-μd, σ2d)

        p = self.differ.as_square(p, .5)
        q = self.differ.as_square(q, .5)

        # Don't need this for the optimization, but making this 
        # interpretable as an actual logprob is useful for debugging
        log_n_choose_w = torch.lgamma(n.float()+1) - torch.lgamma(w.float()+1) - torch.lgamma((n-w).float()+1)
 
        return log_n_choose_w + w*p + (n - w)*q

    def expected_prior(self):
        # Don't need this for the optimization, but making this 
        # interpretable as an actual logprob is useful for debugging
        const = -self.N/2*np.log(2*np.pi*σ0**2)

        # Proof:
        # from sympy.stats import E, Normal
        # s, μ, μ0, σ, σ0 = symbols('s μ μ_0 σ σ_0')
        # s = Normal('s', μ, σ)
        # 1/(2*σ0**2)*E(-(s - μ0)**2)
        return const - 1/(2*σ0**2)*((self.μ - μ0)**2 + torch.diag(self.Σ))

    def cross_entropy(self, n, w):
        return -self.expected_log_likelihood(n, w).sum() - self.expected_prior().sum()

    def entropy(self):
        return 1/2*(self.N*np.log(2*np.pi*np.e) + torch.logdet(self.Σ))

    def forward(self, n, w):
        return -self.cross_entropy(n, w) + self.entropy()

class Solver:

    def __init__(self, N, tol=.1, T=100):
        self.elbo = ELBO(N)
        self.differ = Differ(N)
        self.tol = tol
        self.T = T

    def __call__(self, n, w):
        optim = torch.optim.LBFGS(self.elbo.parameters())
        trace = []
        for t in range(self.T):

            def closure():
                l = -self.elbo(n, w)
                optim.zero_grad()
                l.backward()
                if torch.isnan(l):
                    import aljpy; aljpy.extract()

                grads = [p.grad for p in self.elbo.parameters()]
                paramnorm = torch.cat([p.data.flatten() for p in self.elbo.parameters()]).pow(2).mean().pow(.5)
                gradnorm = torch.cat([g.flatten() for g in grads]).pow(2).mean().pow(.5)
                relnorm = gradnorm/paramnorm

                print(relnorm)

                trace.append(arrdict.arrdict(
                    l=l.detach(),
                    gradnorm=gradnorm,
                    relnorm=relnorm,
                    Σ=self.elbo.Σ.clone()))

                return l

            #TODO: Use LBFGS's terminate params
            optim.step(closure)
            closure()

            if trace[-1].relnorm < self.tol:
                break

        else:
            print('Didn\'t converge')

        μd, σ2d = map(self.differ.as_square, self.differ(self.elbo.μ, self.elbo.Σ))
        return arrdict.arrdict(
            μ=self.elbo.μ.clone(), 
            Σ=self.elbo.Σ.clone(), 
            μd=μd,
            σd=σ2d**.5,
            trace=arrdict.stack(trace)).detach().numpy()

def plot(soln):
    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches(20, 5)

    ax = axes[0]
    ax.plot(soln.trace.l)
    ax.set_xlim(0, len(soln.trace.l)-1)
    ax.set_title('loss')

    ax = axes[1]
    ax.plot(soln.trace.relnorm)
    ax.set_xlim(0, len(soln.trace.relnorm)-1)
    ax.set_yscale('log')
    ax.set_title('norms')

    ax = axes[2]
    ax.plot(soln.μ)
    ax.set_xlim(0, len(soln.μ)-1)
    ax.set_title('μ')

    ax = axes[3]
    ax.imshow(soln.σd)
    ax.set_title('σd')

def sensitivities(Σ):
    # Residual v. the average agent
    Σ = Σ - np.outer(Σ.mean(0), Σ.mean(0))/Σ.mean()

    # Trace if we conditioned I-J out
    σ2d = np.diag(Σ)[:, None] + np.diag(Σ)[None, :] - 2*Σ
    colerr = ((Σ[:, None, :] - Σ[None, :, :])**2).sum(-1)
    trace = np.zeros_like(Σ)
    np.divide(colerr, σ2d, out=trace, where=(σ2d > 0))

    return trace

def alphas(μd, σd, k):
    dϕ = np.exp(-μd)/(1 + np.exp(-μd))**2

    α = 1/(1 + k*dϕ*σd**2)

    return α

def suggest(soln, k):
    improvement = (1 - alphas(soln.μd, soln.σd, k))*sensitivities(soln.Σ)
    idx = improvement.argmax()
    return np.unravel_index(idx, improvement.shape) 

def test_artificial():
    N = 5

    s = torch.randn(N)

    n = torch.randint(1, 10, (N, N))

    d = s[:, None] - s[None, :]
    w = torch.distributions.Binomial(n, 1/(1 + np.exp(-d))).sample()

    soln = Solver(N)(n, w)

    plt.scatter(s, soln.μ)

def test_organic():
    from scalinglaws.arena import database

    run_name = '2020-11-27 21-32-59 az-test'
    winrate = database.symmetric_winrate(run_name).fillna(0).values
    n = database.symmetric_games(run_name).values
    w = (winrate*n).astype(int)

    n, w = map(torch.as_tensor, (n, w))

    soln = Solver(n.shape[0])(n, w)

    plot(soln)

def test_normal_expectation():
    μ = torch.tensor(1.)
    σ = torch.tensor(2.)

    expected = μ

    expectation = normal_expectation(lambda x: x)
    actual = expectation(μ, σ)

    torch.testing.assert_allclose(expected, actual)