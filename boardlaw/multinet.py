import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from rebar import recurrence, profiling
import time

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

class Streamed(nn.Module):

    def __init__(self, n_features, n_models, n_layers):
        super().__init__()
        self.models = nn.ModuleList([
            nn.ModuleList([nn.Linear(n_features, n_features) for _ in range(n_layers)])
            for _ in range(n_models)])
        self.n_features = n_features
        self.streams = [torch.cuda.Stream() for _ in range(n_models)]

    @profiling.nvtx
    def forward(self, x, slices):
        y = x.new_empty((*x.shape[:-1], self.n_features))
        for s, ls, stream in zip(slices, self.models, self.streams):
            with torch.cuda.stream(stream):
                xs = x[s]
                for l in ls:
                    xs = l(xs)
                    xs = F.relu(xs)
                y[s] = xs
        return y

@profiling.profilable
def benchmark(cls, features=256, layers=8, models=8, envs=64*1024, T=64, device='cuda'):
    assert envs % models == 0

    x = torch.zeros((envs, features), device=device)
    chunk = envs//models
    slices = [slice(i, i+chunk) for i in range(0, envs, chunk)]

    model = cls(features, models, layers).to(device)

    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(T):
            model(x, slices=slices)
        torch.cuda.synchronize()
        end = time.time()
    return 1e6*(end - start)/(T*envs)

def profile(**kwargs):
    #TODO: Check multilayer, check backprop
    results = {}
    for cls in [Optimal, Naive, Streamed]:
        results[cls.__name__] = benchmark(cls, **kwargs)

    df = pd.Series(results)
    print(df)
