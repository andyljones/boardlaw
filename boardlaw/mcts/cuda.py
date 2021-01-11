import torch
import torch.cuda
from .. import cuda
from rebar import profiling

_cache = None
def module():
    global _cache
    if _cache is None:
        _cache = cuda.load(__package__)
    return _cache 

@profiling.nvtx
def mcts(logits, w, n, c_puct, seats, terminal, children):
    B, T, A = logits.shape
    S = w.shape[-1]
    cuda.assert_shape(w, (B, T, S))
    cuda.assert_shape(n, (B, T))
    cuda.assert_shape(c_puct, (B,))
    cuda.assert_shape(seats, (B, T))
    cuda.assert_shape(terminal, (B, T))
    cuda.assert_shape(children, (B, T, A))
    assert (c_puct > 0.).all(), 'Zero c_puct not supported; will lead to an infinite loop in the kernel'
    assert len({logits.device, w.device, n.device, c_puct.device, seats.device, terminal.device, children.device}) == 1, 'Inputs span multiple devices'

    return module().MCTS(logits, w, n, c_puct, seats.short(), terminal, children)

@profiling.nvtx
def Backup(*args, **kwargs):
    return module().Backup(*args, **kwargs)

@profiling.nvtx
def descend(*args, **kwargs):
    return module().descend(*args, **kwargs)

@profiling.nvtx
def root(*args, **kwargs):
    return module().root(*args, **kwargs)

@profiling.nvtx
def backup(*args, **kwargs):
    return module().backup(*args, **kwargs)
