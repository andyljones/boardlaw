import pandas as pd
import aljpy
import torch
from torch import nn
from torch.nn import functional as F
from rebar import recurrence

class Naive(nn.Module):

    def __init__(self, in_features, out_features, n_models):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(n_models)])
        self.out_features = out_features

    def forward(self, x, idxs):
        y = x.new_empty((*x.shape[:-1], self.out_features))
        for i, l in enumerate(self.layers):
            y[idxs == i] = l(x[idxs == i])
        return F.relu(y)

class Streamed(nn.Module):

    def __init__(self, in_features, out_features, n_models):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(n_models)])
        self.out_features = out_features
        self.streams = [torch.cuda.Stream() for _ in range(n_models)]

    def forward(self, x, idxs):
        y = x.new_empty((*x.shape[:-1], self.out_features))
        for i, l in enumerate(self.layers):
            with torch.cuda.stream(self.streams[i]):
                y[idxs == i] = l(x[idxs == i])
        return F.relu(y)

def benchmark(cls, features=128, layers=1, models=1, envs=8192, T=128, device='cuda:1'):
    assert envs % models == 0

    x = torch.zeros((envs, features), device=device)
    idxs = torch.arange(envs, device=device) % models

    model = []
    for _ in range(layers):
        model.extend([cls(features, features, models)])
    model = recurrence.Sequential(*model).to(device)

    with aljpy.timer() as timer:
        torch.cuda.synchronize()
        for t in range(T):
            y = model(x, idxs=idxs)
        torch.cuda.synchronize()
    return 1e6*timer.time()/(T*envs)

def profile(**kwargs):
    results = {}
    for cls in [Naive, Streamed]:
        results[cls.__name__] = pd.Series({m: benchmark(cls, **kwargs, models=m) for m in [1, 2, 4, 8, 16, 32, 64]})
    return pd.concat(results, 1)