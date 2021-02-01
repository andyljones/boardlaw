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
