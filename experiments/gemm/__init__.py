import time
import torch
from boardlaw import cuda

_cache = None
def module():
    global _cache
    if _cache is None:
        _cache = cuda.load(__package__, files=('wrappers.cpp',))
    return _cache 

def test_dims():
    B = 5
    U, V = 2, 3

    W = torch.zeros((1, U, V)).cuda()
    x = torch.zeros((1, V)).cuda()
    b = torch.zeros((1, U)).cuda()

    idxs = torch.zeros((B,)).long().cuda()

    y = module().linear(W, x, b, idxs)

    y.cpu()

def test_one_batch():
    B = 5
    U, V = 2, 3

    W = torch.as_tensor([[[2.]]]).float().cuda()
    x = torch.as_tensor([[2.]]).float().cuda()
    b = torch.as_tensor([[1.]]).float().cuda()

    idxs = torch.as_tensor([0]).long().cuda()

    y = module().linear(W, x, b, idxs)

    torch.testing.assert_allclose(y.cpu(), [[5]])

def benchmark():
    M = 16
    B = 16*1024
    U = 256
    V = 256
    T = 128

    W = torch.zeros((M, U, V)).cuda()
    x = torch.zeros((B, V)).cuda()
    b = torch.zeros((M, U)).cuda()
    idxs = torch.arange(B).cuda() % M

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(T):
        module().linear(W, x, b, idxs)

    torch.cuda.synchronize()
    end = time.time()

    print((B*T)/(end - start)/1e6)

def ideal():
    M = 16
    B = 16*1024
    U = 256
    V = 256
    T = 128

    W = torch.zeros((U, V)).cuda()
    x = torch.zeros((B, V)).cuda()
    b = torch.zeros((U,)).cuda()
    idxs = torch.arange(B).cuda() % M

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(T):
        torch.matmul(W, x.unsqueeze(-1)).squeeze(-1) + b

    torch.cuda.synchronize()
    end = time.time()

    print((B*T)/(end - start)/1e6)

def sequential():
    M = 16
    B = 16*1024
    U = 256
    V = 256
    T = 128

    W = torch.zeros((M, U, V)).cuda()
    x = torch.zeros((B, V)).cuda()
    b = torch.zeros((M, U,)).cuda()
    idxs = torch.arange(B).cuda() % M

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(T):
        for m in range(M):
            torch.matmul(W[m], x[idxs == m].unsqueeze(-1)).squeeze(-1) + b[m]

    torch.cuda.synchronize()
    end = time.time()

    print((B*T)/(end - start)/1e6)