import pandas as pd
import torch
import time

def ideal(X, W, idxs, streams, repeats):
    for _ in range(repeats):
        X = X @ W[0]

def naive(X, W, idxs, streams, repeats):
    for i, w in enumerate(W):
        for _ in range(repeats):
            X[idxs == i] = X[idxs == i] @ w
    
def streamed(X, W, idxs, streams, repeats):
    for i, (w, s) in enumerate(zip(W, streams)):
        with torch.cuda.stream(s):
            for _ in range(repeats):
                X[idxs == i] = X[idxs == i] @ w

def broadcast(X, W, idxs, streams, repeats):
    for _ in range(repeats):
        X[:] = torch.bmm(X[:, None, :], W[idxs]).squeeze(1)

def benchmark(batchsize=64*1024, n_models=32, dim=128, n_repeats=32):
    assert batchsize % n_models == 0
    X = torch.zeros((batchsize, dim)).cuda()
    W = torch.zeros((n_models, dim, dim)).cuda()

    streams = [torch.cuda.Stream() for _ in range(n_models)]
    idxs = torch.arange(batchsize).cuda() % n_models

    times = {}
    for f in [ideal, naive, streamed, broadcast]:    
        torch.cuda.synchronize()
        start = time.time()

        times[f.__name__] = f(X, W, idxs, streams, n_repeats)

        torch.cuda.synchronize()
        end = time.time()

        times[f.__name__] = end - start

    times = pd.Series(times).mul(1000)
    return times.round(2)