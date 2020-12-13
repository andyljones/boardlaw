import torch
import torch.cuda
from .. import cuda

loaded = cuda.load(__package__)
for k in dir(loaded):
    if not k.startswith('__'):
        globals()[k] = getattr(loaded, k)

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

    return loaded.MCTS(logits, w, n.int(), c_puct, seats.int(), terminal, children.int())