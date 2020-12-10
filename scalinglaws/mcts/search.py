import pickle
import numpy as np
import torch
import torch.utils.cpp_extension
from rebar import arrdict
import sysconfig
from pkg_resources import resource_filename

DEBUG = True

def _cuda():
    [torch_libdir] = torch.utils.cpp_extension.library_paths()
    python_libdir = sysconfig.get_config_var('LIBDIR')
    libpython_ver = sysconfig.get_config_var('LDVERSION')
    return torch.utils.cpp_extension.load(
        name='scalinglawscuda', 
        sources=[resource_filename(__package__, f'cpp/{fn}') for fn in ('wrappers.cpp', 'kernels.cu')], 
        extra_cflags=['-std=c++17'] + (['-g'] if DEBUG else []), 
        extra_cuda_cflags=['--use_fast_math', '-lineinfo', '-std=c++14'] + (['-g', '-G'] if DEBUG else []),
        extra_ldflags=[
            f'-lpython{libpython_ver}', '-ltorch', '-ltorch_python', '-lc10_cuda', '-lc10', 
            f'-L{torch_libdir}', f'-Wl,-rpath,{torch_libdir}',
            f'-L{python_libdir}', f'-Wl,-rpath,{python_libdir}'])

cuda = _cuda()

def safe_div(x, y):
    r = x/y
    r[x == 0] = 0
    return r

def newton_search(f, grad, x0, tol=1e-3):
    # Some guidance on what's going on here:
    # * While the Regularized MCTS paper recommends binary search, it turns out to be pretty slow and - thanks
    #   to the numerical errors that show up when you run this whole thing in float32 - tricky to implement.
    # * What works better is to exploit the geometry of the problem. The error function is convex and 
    #   descends from an asymptote somewhere to the left of x0. 
    # * By taking Newton steps from x0, we head right and so don't run into any more asymptotes.
    # * More, because the error is convex, Newton's is guaranteed to undershoot the solution.
    # * The only thing we need to be careful about is when we start near the asymptote. In that case the gradient
    #   is really large and it's possible that - again, thanks to numerical issues - our steps 
    #   *won't actually change the error*. I couldn't think of a good solution to this, but so far in practice it
    #   turns out 'just giving up' works pretty well. It's a rare occurance - 1 in 40k samples of the benchmark
    #   run I did - and 'just give up' only missed the specified error tol by a small amount.
    x = x0.clone()
    y = torch.zeros_like(x)
    steps = 0
    while True:
        y_new = f(x)
        # Gonna tolerate the occasional negative y value here, cause we're not gonna be able to fix it with this
        # scheme.
        done = (y_new < tol) | (y == y_new)
        if done.all():
            return x, steps
        y = y_new

        x[~done] = (x - y/grad(x))[~done]
        steps += 1

def solve_policy_old(pi, q, lambda_n):
    assert (lambda_n > 0).all(), 'Don\'t currently support zero lambda_n'

    # Need alpha_min to be at least 2eps greater than the asymptote, else we'll risk an infinite gradient
    eps = torch.finfo(torch.float).eps
    gap = (lambda_n[:, None]*pi).clamp(2*eps, None)
    alpha_min = (q + gap).max(-1).values

    policy = lambda alpha: safe_div(lambda_n[:, None]*pi, alpha[:, None] - q)
    error = lambda alpha: policy(alpha).sum(-1) - 1
    grad = lambda alpha: -safe_div(lambda_n[:, None]*pi, (alpha[:, None] - q).pow(2)).sum(-1)

    alpha_star, steps = newton_search(error, grad, alpha_min)

    p = policy(alpha_star)

    return arrdict.arrdict(
        policy=p,
        steps=steps,
        alpha_min=alpha_min, 
        alpha_star=alpha_star,
        error=p.sum(-1) - 1)

def solve_policy(pi, q, lambda_n):
    assert (lambda_n > 0).all(), 'Don\'t currently support zero lambda_n'

    with torch.cuda.device(pi.device):
        alpha_star = cuda.solve_policy(pi, q, lambda_n)
    p = lambda_n[:, None]*pi/(alpha_star[:, None] - q)

    return arrdict.arrdict(
        policy=p,
        alpha_star=alpha_star,
        error=p.sum(-1) - 1)

def test_policy():
    # Case when the root is at the lower bound
    pi = torch.tensor([[.999, .001]])
    q = torch.tensor([[0., 1.]])
    lambda_n = torch.tensor([[.1]])
    soln = solve_policy(pi, q, lambda_n)
    torch.testing.assert_allclose(soln.alpha_star, torch.tensor([[1.]]), rtol=.001, atol=.001)

    # Case when the root is at the upper bound
    pi = torch.tensor([[.5, .5]])
    q = torch.tensor([[1., 1.]])
    lambda_n = torch.tensor([[.1]])
    soln = solve_policy(pi, q, lambda_n)
    torch.testing.assert_allclose(soln.alpha_star, torch.tensor([[1.1]]), rtol=.001, atol=.001)

    # Case when the root is at the upper bound
    pi = torch.tensor([[.25, .75]])
    q = torch.tensor([[1., .25]])
    lambda_n = torch.tensor([[.5]])
    soln = solve_policy(pi, q, lambda_n)
    torch.testing.assert_allclose(soln.alpha_star, torch.tensor([[1.205]]), rtol=.001, atol=.001)

def test_data(small=False):
    fn = 'output/search/benchmark.pkl' if small else 'output/search/extra-benchmark.pkl' 
    with open(fn, 'rb') as f:
        return pickle.load(f)

def test_cuda():
    pi = torch.tensor([
        [1/3., 2/3.],
        [3/4., 1/4.]]).cuda()
    q = torch.tensor([
        [.3, .1],
        [.5, .7]]).cuda()
    lambda_n = torch.tensor([1., 2.]).cuda()

    print('Runnign solve')
    soln = solve_policy(pi, q, lambda_n)

    return print(soln.error)
    
def benchmark_search(T=500):
    import aljpy

    args = [a.cuda() for a in test_data()]

    np.random.seed(0)
    assert T < len(args)
    args = np.random.permutation(args)[:T]

    solns = []
    torch.cuda.synchronize()
    with aljpy.timer() as timer:
        for i, arg in enumerate(args):
            solns.append(solve_policy(**arg))
        torch.cuda.synchronize()
    solns = arrdict.cat(solns)
    args = arrdict.cat(args)
    t = 1000*timer.time()
    print(f'{t:.0f}ms total over {T} policies')
    print(f'{t/T:.2f}ms/policy')
    print(f'{solns.error.abs().mean():.3f} MAD')

