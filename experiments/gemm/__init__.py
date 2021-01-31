import torch
from boardlaw import cuda

_cache = None
def module():
    global _cache
    if _cache is None:
        _cache = cuda.load(__package__, files=('wrappers.cpp',))
    return _cache 

def test():
    B = 5
    U, V = 2, 3

    W = torch.zeros((B, U, V)).cuda()
    x = torch.zeros((B, V)).cuda()
    b = torch.zeros((B, U)).cuda()

    idxs = torch.zeros((B,)).int().cuda()

    y = module().linear(W, x, b, idxs)