import pandas as pd
from rebar import arrdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions
import torch.testing
from torch import nn
import geotorch
from . import expectations, common
from logging import getLogger

log = getLogger(__name__)

μ0 = 0
σ0 = 10

def pairwise_indices(N):
    j, k = torch.as_tensor(np.indices((N, N)).reshape(2, -1))
    j, k = j[j != k], k[j != k]
    return j, k

def pairwise_diffs(μ, Σ):
    j, k = pairwise_indices(len(μ))

    μd = μ[j] - μ[k]
    σ2d = Σ[j, j] - Σ[j, k] - Σ[k, j] + Σ[k, k]

    return μd, σ2d

def as_square(xd, fill=0.):
    N = int((1 + (1 + 4*len(xd))**.5)/2)
    j, k = pairwise_indices(N)
    y = torch.full((N, N), fill).float()
    y[j, k] = xd
    return y

_cache = None
class ELBO(nn.Module):
    
    def __init__(self, N, constrain=True):
        super().__init__()
        self.N = N
        self.register_parameter('μ', nn.Parameter(torch.zeros((N,)).float()))
        self.register_parameter('Σ', nn.Parameter(torch.eye(N).float()))
        # Useful to be able to turn this off for testing
        if constrain:
            geotorch.positive_definite(self, 'Σ')

        # This is expensive to construct, so let's cache it
        global _cache
        if _cache is None:
            _cache = expectations.normal(lambda d: -np.log(1 + np.exp(-d))) 
        self.expectation = _cache

    def expected_prior(self):
        # Constant isn't strictly needed, but it does help with testing
        const = -1/2*np.log(2*np.pi) - np.log(σ0)

        return const - 1/(2*σ0**2)*((self.μ - μ0)**2 + torch.diag(self.Σ))

    def expected_log_likelihood(self, n, w):
        # Constant isn't strictly needed, but it does help with testing
        const = torch.lgamma(n.float()+1) - torch.lgamma(w.float()+1) - torch.lgamma((n-w).float()+1)

        μd, σ2d = pairwise_diffs(self.μ, self.Σ)

        p = self.expectation(μd, σ2d)
        q = self.expectation(-μd, σ2d)

        p = as_square(p, -np.log(2))
        q = as_square(q, -np.log(2))
 
        return const + w*p + (n - w)*q

    def cross_entropy(self, n, w):
        return -self.expected_prior().sum() - self.expected_log_likelihood(n, w).sum() 

    def entropy(self):
        if torch.logdet(self.Σ) < 0:
            raise ValueError('Σ has become negdef')
        return 1/2*(self.N*np.log(2*np.pi*np.e) + torch.logdet(self.Σ))

    def forward(self, n, w):
        return -self.cross_entropy(n, w) + self.entropy()

class Solver:

    def __init__(self, N, **kwargs):
        self.N = N
        self.kwargs = {'max_iter': 100, **kwargs}

    def __call__(self, n, w):
        n = torch.as_tensor(n)
        w = torch.as_tensor(w)
        elbo = ELBO(self.N)

        # The gradients around here can be a little explode-y; a line search is a bit slow but 
        # keeps us falling up any cliffs.
        optim = torch.optim.LBFGS(
            elbo.parameters(), 
            line_search_fn='strong_wolfe', 
            **self.kwargs)

        trace = []

        def closure():
            l = -elbo(n, w)
            if torch.isnan(l):
                raise ValueError('Hit a nan.')
            optim.zero_grad()
            l.backward()

            grads = [p.grad for p in elbo.parameters()]
            paramnorm = torch.cat([p.data.flatten() for p in elbo.parameters()]).pow(2).mean().pow(.5)
            gradnorm = torch.cat([g.flatten() for g in grads]).pow(2).mean().pow(.5)
            relnorm = gradnorm/paramnorm

            trace.append(arrdict.arrdict(
                l=l,
                gradnorm=gradnorm,
                relnorm=relnorm,
                Σ=elbo.Σ).detach().clone())

            return l

        try:
            optim.step(closure)
            closure()
        except ValueError as e:
            log.warn(f'activelo did not converge: "{str(e)}"')

        μd, σ2d = map(as_square, pairwise_diffs(elbo.μ, elbo.Σ))
        return arrdict.arrdict(
            n=n,
            w=w,
            μ=elbo.μ, 
            Σ=elbo.Σ, 
            μd=μd,
            σd=σ2d**.5,
            trace=arrdict.stack(trace)).detach().numpy()

def solve(n, w):
    if isinstance(n, pd.DataFrame):
        return arrdict.arrdict({k: common.pandify(v, n.index) for k, v in solve(n.values, w.values).items()})
    return Solver(n.shape[0])(n, w)

def test_elbo():
    elbo = ELBO(2, constrain=False)
    elbo.μ[:] = torch.tensor([1., 2.])
    elbo.Σ[:] = torch.tensor([[1., .5], [.5, 2.]])

    approx = torch.distributions.MultivariateNormal(elbo.μ, elbo.Σ)
    s = approx.sample((100000,))

    # Test entropy
    expected = -approx.log_prob(s).mean()
    torch.testing.assert_allclose(expected, elbo.entropy(), rtol=.01, atol=.01)

    # Test prior
    prior = torch.distributions.MultivariateNormal(torch.full((2,), μ0), σ0**2 * torch.eye(2))
    expected = prior.log_prob(s).mean()
    torch.testing.assert_allclose(expected, elbo.expected_prior().sum(), rtol=.01, atol=.01)

    # Test likelihood
    n = torch.tensor([[0, 3], [3, 0]])
    w = torch.tensor([[0, 1], [1, 0]])

    s = torch.distributions.MultivariateNormal(elbo.μ, elbo.Σ).sample((100000,))
    d = s[:, :, None] - s[:, None, :]
    r = 1/(1 + torch.exp(-d))
    log_likelihood = torch.distributions.Binomial(n, r).log_prob(w.float())

    expected = log_likelihood.mean(0)

    torch.testing.assert_allclose(expected, elbo.expected_log_likelihood(n, w), rtol=.01, atol=.01)

def test_solver():
    #TODO: Should Monte-Carlo the posterior to a small problem and check the KL-div from 
    # it to the approx'd posterior
    pass
