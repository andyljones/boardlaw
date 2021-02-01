import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from rebar import recurrence, profiling
import time
from . import cuda

class Optimal(nn.Module):

    def __init__(self, features, n_models, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(features, features) for _ in range(n_layers)])

    @profiling.nvtx
    def forward(self, x, slices):
        for l in self.layers:
            x = l(x)
            x = F.relu(x)
        return x

class Naive(nn.Module):

    def __init__(self, n_features, n_models, n_layers):
        super().__init__()
        self.models = nn.ModuleList([
            nn.ModuleList([nn.Linear(n_features, n_features) for _ in range(n_layers)])
            for _ in range(n_models)])
        self.n_features = n_features

    @profiling.nvtx
    def forward(self, x, slices):
        y = x.new_empty((*x.shape[:-1], self.n_features))
        for s, ls in zip(slices, self.models):
            xs = x[s]
            for l in ls:
                xs = l(xs)
                xs = F.relu(xs)
            y[s] = xs
        return y

class _Streamed(nn.Module):

    def __init__(self, n_features, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_features, n_features) for _ in range(n_layers)])

    def forward(self, xs):
        for l in self.layers:
            xs = l(xs)
            xs = F.relu(xs)
        return xs

class Streamed(nn.Module):

    def __init__(self, n_features, n_models, n_layers):
        super().__init__()

        exemplar = torch.zeros((1, n_features))
        self.models = nn.ModuleList([torch.jit.trace(_Streamed(n_features, n_layers), (exemplar,)) for _ in range(n_models)])
        self.n_features = n_features
        self.streams = [torch.cuda.Stream() for _ in range(n_models)]

    @profiling.nvtx
    def forward(self, x, slices):
        parts = []
        for s, model, stream in zip(slices, self.models, self.streams):
            with torch.cuda.stream(stream):
                parts.append(model(x[s]))
        torch.cuda.synchronize()
        return torch.cat(parts)

class Compiled(nn.Module):

    def __init__(self, n_features, n_models, n_layers):
        super().__init__()

        self.Ws = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.zeros(n_features, n_features)) for _ in range(n_layers)]) for _ in range(n_models)])
        self.bs = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.zeros(n_features,)) for _ in range(n_layers)]) for _ in range(n_models)])

        self._forward = cuda.load(__package__).forward

    @profiling.nvtx
    def forward(self, x, slices):
        return self._forward(x, self.Ws, self.bs, [(s.start, s.stop) for s in slices])

class Hybrid(nn.Module):

    def __init__(self, n_features, n_models, n_layers):
        super().__init__()
        self.n_models = n_models
        self.n_layers = n_layers

        self.register_parameter('Wbig', nn.Parameter(torch.zeros((n_layers, n_features, n_features))))
        self.register_parameter('Bbig', nn.Parameter(torch.zeros((n_layers, n_features,))))

        self.register_parameter('Wlil', nn.Parameter(torch.zeros((n_layers, n_models-1, n_features, n_features))))
        self.register_parameter('Blil', nn.Parameter(torch.zeros((n_layers, n_models-1, n_features))))

        self._forward = cuda.load(__package__).forward
        self.bigstream = torch.cuda.Stream()
        self.lilstream = torch.cuda.Stream()

    @profiling.nvtx
    def forward(self, x, slices):
        split = slices[0].stop
        with torch.cuda.stream(self.bigstream):
            ybig = x[:split]
            for l in range(self.n_layers):
                ybig = torch.addmm(self.Bbig[l], ybig, self.Wbig[l])
        
        n_lil = slices[1].stop - slices[1].start
        with torch.cuda.stream(self.lilstream):
            ylil = x[split:]
            for l in range(self.n_layers):
                Wlil = self.Wlil[l].repeat_interleave(n_lil, 0)
                Blil = self.Blil[l].repeat_interleave(n_lil, 0)
                ylil = torch.baddbmm(Blil[..., None], Wlil, ylil[..., None]).squeeze(-1)

        return torch.cat([ybig, ylil])


@profiling.profilable
@profiling.nvtx
def benchmark(cls, features=256, layers=8, models=4, envs=8*1024, T=8, device='cuda'):
    assert envs % models == 0

    x = torch.zeros((envs, features), device=device)
    chunk = envs//4//models
    start = 3*envs//4
    slices = [slice(0, start)] + [slice(i, i+chunk) for i in range(start, envs, chunk)]

    model = cls(features, models+1, layers).to(device)

    # Warm up
    model(x, slices=slices)

    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(T):
            model(x, slices=slices)
        torch.cuda.synchronize()
        end = time.time()
    return 1e6*(end - start)/(T*envs)

def profile(reps=1, **kwargs):
    """
    CUDA_VISIBLE_DEVICES=1 nsys profile --force-overwrite true -o "output/nsys" -c cudaProfilerApi --stop-on-range-end false -t cuda,cublas,nvtx -e EMIT_NVTX=1 python -c "from boardlaw.multinet import *; profile()"
    """
    torch.tensor(1).cuda() # initialize torch outside profiler range
    torch.cuda.synchronize()
    results = {}
    for _ in range(reps):
        for cls in [Optimal, Naive, Streamed, Compiled, Hybrid]:
            results.setdefault(cls.__name__, []).append(benchmark(cls, **kwargs))

    s = pd.DataFrame(results).mean()
    print(s)
    return s

def curve(**kwargs):
    results = {}
    for m in [1, 2, 4, 8, 16]:
        print(f'running {m}')
        results[m] = profile(models=m, **kwargs)
    results = pd.concat(results, 1)

    return results.div(results.loc['Optimal'], 1).T[['Naive', 'Streamed']].round(2)

