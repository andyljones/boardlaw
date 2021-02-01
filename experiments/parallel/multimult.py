"""I want to do many small matrix multiplications as fast as possible. Below are some possible approaches, 
and a benchmark function. On my RTX 2080 Ti, I get 
```
ideal          8ms
naive        191ms
streamed     188ms
broadcast    551ms
```

* The `ideal` approach is the hypothetical upper bound, if there were only one set of weights being used rather than many.
* The `naive` approach does each multiplication separately, one after another
* The `streamed` approach tries to use CUDA streams to do them all in parallel
* The `broadcast` approach expands the weights to match the batchsize, then uses a batched matrix mult.

I've two questions:
* Why is the `streamed` approach no faster than `naive`?
* Are there any other approaches that get closer to the `ideal` performance?
"""
import pandas as pd
import torch
import time
from rebar import profiling as prof
from boardlaw import cuda

@prof.nvtx
def ideal(X, W, slices, streams, repeats):
    for _ in range(repeats):
        X = X @ W[0]

@prof.nvtx
def naive(X, W, slices, streams, repeats):
    for i, w in enumerate(W):
        for _ in range(repeats):
            X[slices[i]] = X[slices[i]] @ w
    
@prof.nvtx
def streamed(X, W, slices, streams, repeats):
    for i, (w, s) in enumerate(zip(W, streams)):
        with torch.cuda.stream(s):
            for _ in range(repeats):
                X[slices[i]] = X[slices[i]] @ w

@prof.profilable
def benchmark(batchsize=64*1024, n_weights=16, dim=256, n_repeats=16):
    assert batchsize % n_weights == 0
    X = torch.zeros((batchsize, dim)).cuda()
    W = torch.zeros((n_weights, dim, dim)).cuda()

    streams = [torch.cuda.Stream() for _ in range(n_weights)]
    chunk = batchsize // n_weights
    slices = [slice(i, i+chunk) for i in range(0, batchsize, chunk)]

    cstreamed = prof.nvtx(cuda.load(__package__, ('multimult.cpp',)).cstreamed)

    times = {}
    for f in [ideal, naive, streamed, cstreamed]:
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.time()

            if f == cstreamed:
                f(X, W, [s.start for s in slices], [s.stop for s in slices], n_repeats)
            else:
                f(X, W, slices, streams, n_repeats)

            torch.cuda.synchronize()
            end = time.time()

            times.setdefault(f.__name__, []).append(end - start)

    times = batchsize/pd.DataFrame(times).median()
    print(times.map(lambda t: f'{t/1e6:.1f}m samples/sec'))

if __name__ == '__main__':
    with torch.autograd.profiler.emit_nvtx():
        benchmark()