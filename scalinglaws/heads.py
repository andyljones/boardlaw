import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple

Empty = namedtuple('Empty', ())
Discrete = namedtuple('Discrete', ('dim',))
Masked = namedtuple('Masked', ('dim',))
Vector = namedtuple('Vector', ('dim',))
Tensor = namedtuple('Tensor', ('dim',))

class EmptyIntake(nn.Module):

    def __init__(self, space, width):
        super().__init__()

        bias = nn.Parameter(torch.zeros(width))
        self.register_parameter('bias', bias)

    def forward(self, obs, **kwargs):
        if obs.ndim == 2:
            return self.forward(obs[None], **kwargs).squeeze(0)

        T, B, _ = obs.shape
        return self.bias[None, None, :].repeat((T, B, 1))

class VectorIntake(nn.Linear):

    def __init__(self, space, width):
        C, = space.dim
        super().__init__(C, width)
                               
    def forward(self, obs, **kwargs):
        if obs.ndim == 2:
            return self.forward(obs[None], **kwargs).squeeze(0)

        T, B, C = obs.shape
        return super().forward(obs.reshape(T*B, C)).reshape(T, B, -1)

class TensorIntake(nn.Linear):

    def __init__(self, space, width):
        self._ndim = len(space.dim)
        super().__init__(int(np.prod(space.dim)), width)

    def forward(self, obs, *args, **kwargs):
        if obs.ndim == self._ndim+1:
            return self.forward(obs[None], *args, **kwargs).squeeze(0)

        T, B = obs.shape[:2]
        return super().forward(obs.reshape(T*B, -1)).reshape(T, B, -1)

class ConcatIntake(nn.Module):

    def __init__(self, space, width):
        super().__init__()

        intakes = type(space)({k: intake(v, width) for k, v in space.items()})
        self.core = nn.Linear(len(intakes)*width, width)
        self.intakes = nn.ModuleDict(intakes)

    def forward(self, x, **kwargs):
        ys = [self.intakes[k](x[k]) for k in self.intakes]
        return self.core(torch.cat(ys, -1))

def intake(space, width):
    if isinstance(space, dict):
        return ConcatIntake(space, width)
    name = f'{type(space).__name__}Intake'
    if name in globals():
        return globals()[name](space, width)
    raise ValueError(f'Can\'t handle {space}')

class DiscreteOutput(nn.Module):

    def __init__(self, space, width):
        super().__init__()
        shape = space
        self.core = nn.Linear(width, int(np.prod(shape)))
        self.shape = shape
    
    def forward(self, x, **kwargs):
        y = self.core(x).reshape(*x.shape[:-1], *self.shape)
        return F.log_softmax(y, -1)

    def sample(self, logits, test=False):
        if test:
            return logits.argmax(-1)
        else:
            return torch.distributions.Categorical(logits=logits).sample()

class MaskedOutput(nn.Module):

    def __init__(self, space, width):
        super().__init__()
        shape = space
        self.core = nn.Linear(width, int(np.prod(shape)))
        self.shape = shape
    
    def forward(self, x, valid, *args, **kwargs):
        y = self.core(x).reshape(*x.shape[:-1], *self.shape)
        y = y.where(valid, torch.full_like(y, -np.inf))
        return F.log_softmax(y, -1)

    def sample(self, logits, test=False):
        if test:
            return logits.argmax(-1)
        else:
            return torch.distributions.Categorical(logits=logits).sample()

class DictOutput(nn.Module):

    def __init__(self, space, width):
        super().__init__()
        self.core = nn.Linear(width, width*len(space))

        self._dtype = type(space)
        self.outputs = nn.ModuleDict({k: output(v, width) for k, v in space.items()})

    def forward(self, x, **kwargs):
        ys = torch.chunk(self.core(x), len(self.outputs), -1)
        return self._dtype({k: v(ys[i]) for i, (k, v) in enumerate(self.outputs.items())})
    
    def sample(self, l):
        return self._dtype({k: v.sample(l[k]) for k, v in self.outputs.items()})

def scatter_values(v, seats):
    seats = torch.stack([seats, 1-seats], -1)
    vs = torch.stack([v, -v], -1)
    xs = torch.full_like(vs, np.nan)
    return xs.scatter(-1, seats.long(), vs)

class ValueOutput(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.core = nn.Linear(width, 1)

    def forward(self, x, valid, seats, *args, **kwargs):
        v = self.core.forward(x).squeeze(-1)
        return scatter_values(v, seats)

def output(space, width):
    if isinstance(space, dict):
        return DictOutput(space, width)
    name = f'{type(space).__name__}Output'
    if name in globals():
        return globals()[name](space, width)
    raise ValueError(f'Can\'t handle {space}')