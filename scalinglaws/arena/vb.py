import numpy as np
import sympy as sym
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

μ0 = 0
σ0 = 1

μ_lims = [-5, +5]
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

class LUT:

    def __init__(self, f, N, K=101, S=1000):
        self.μ_range = np.linspace(*μ_lims, K)
        self.σ_range = np.logspace(*σ_lims, K, base=10)

        #TODO: Importance sample these zs
        zs = np.linspace(-5, +5, S)
        pdf = sp.stats.norm.pdf(zs)[None, None, :]
        ds = (self.μ_range[:, None, None] + zs[None, None, :]*self.σ_range[None, :, None])
        self.expectation = (f(ds)*pdf/pdf.sum()).sum(-1)
        self.interp = sp.interpolate.RectBivariateSpline(self.μ_range, self.σ_range, self.expectation, kx=1, ky=1)

        j, k = np.indices((N, N)).reshape(2, -1)
        row = np.arange(len(j))
        R = np.zeros((len(row), N))
        R[row, j] = 1
        R[row, k] = -1
        self.R = R

    def winrates(self, μ, Σ):
        μd = self.R @ μ
        σd = np.diag(self.R @ Σ @ self.R.T)**.5

        N = self.R.shape[1]
        return self.interp(μd, σd, grid=False).reshape(N, N)

    def plot(self):
        Y, X = np.meshgrid(self.μ_range, self.σ_range)
        (t, b), (l, r) = μ_lims, σ_lims
        plt.imshow(np.exp(self.expectation), extent=(l, r, b, t), vmin=0, vmax=1, cmap='RdBu', aspect='auto')
        plt.colorbar()

def expected_log_likelihood(n, w, μ, Σ):
    self = expected_log_likelihood
    if not hasattr(self, '_lut'):
        self._lut = LUT(lambda d: -np.log(1 + np.exp(-d)), n.shape[0])
    lut = self._lut

    return w*lut.winrates(μ, Σ) + (n - w)*lut.winrates(-μ, Σ)

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
