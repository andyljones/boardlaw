import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from rebar import recurrence, timer

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
        #TODO: This doesn't work at all
        y = x.new_empty((*x.shape[:-1], self.out_features))
        torch.cuda.synchronize()
        for i, l in enumerate(self.layers):
            with torch.cuda.stream(self.streams[i]):
                y[idxs == i] = l(x[idxs == i])
        torch.cuda.synchronize()
        return F.relu(y)

class AddMM(nn.Module):

    def __init__(self, in_features, out_features, n_models):
        #TODO: Why is this slower with one model than BMM?
        super().__init__()
        self.register_parameter('w', nn.Parameter(torch.zeros((n_models, in_features, out_features))))
        self.register_parameter('b', nn.Parameter(torch.zeros((n_models, out_features))))
        self.out_features = out_features

    def forward(self, x, idxs):
        y = x.new_empty((*x.shape[:-1], self.out_features))
        for i in range(self.w.size(0)):
            y[idxs == i] = torch.addmm(self.b[i], x[idxs == i], self.w[i])
        return F.relu(y)

class BAddBMM(nn.Module):

    def __init__(self, in_features, out_features, n_models):
        super().__init__()
        self.register_parameter('w', nn.Parameter(torch.zeros((n_models, in_features, out_features))))
        self.register_parameter('b', nn.Parameter(torch.zeros((n_models, out_features))))

    def forward(self, x, idxs):
        y = torch.baddbmm(self.b[idxs, :, None], self.w[idxs], x[:, :, None]).squeeze(-1)
        return F.relu(y)

def benchmark(cls, features=128, layers=1, models=1, envs=8192, T=128, device='cuda:1'):
    assert envs % models == 0

    x = torch.zeros((envs, features), device=device)
    idxs = torch.arange(envs, device=device) % models

    model = []
    for _ in range(layers):
        model.extend([cls(features, features, models)])
    model = recurrence.Sequential(*model).to(device)

    with timer.timer(cuda=True) as t:
        for _ in range(T):
            y = model(x, idxs=idxs)
    return 1e6*t.time()/(T*envs)

def profile(**kwargs):
    #TODO: Check multilayer, check backprop
    results = {}
    for cls in [AddMM, Naive, Streamed, BAddBMM]:
        results[cls.__name__] = pd.Series({m: benchmark(cls, **kwargs, models=m) for m in [1, 2, 4, 8, 16, 32]})

    df = pd.concat(results, 1)
    ax = df.plot(logx=True, logy=True, marker='o')
    ax.set_ylabel('μs/sample')
    ax.set_xlabel('n models')
    return df
