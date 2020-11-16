import numpy as np
import torch
from . import heads, lstm
from torch import nn
from rebar import recurrence, arrdict
from torch.nn import functional as F

class Residual(nn.Linear):

    def __init__(self, width):
        super().__init__(width, width)

    def forward(self, x, **kwargs):
        return x + F.relu(super().forward(x))

def scatter_values(v, seats):
    seats = torch.stack([seats, 1-seats], -1)
    vs = torch.stack([v, -v], -1)
    xs = torch.full_like(vs, np.nan)
    return xs.scatter(-1, seats.long(), vs)

class Network(nn.Module):

    def __init__(self, obs_space, action_space, width=256):
        super().__init__()
        out = heads.output(action_space, width)
        self.sampler = out.sample
        self.policy = recurrence.Sequential(
            heads.intake(obs_space, width),
            Residual(width),
            Residual(width),
            # lstm.LSTM(width),
            out)
        self.value = recurrence.Sequential(
            heads.intake(obs_space, width),
            Residual(width),
            Residual(width),
            # lstm.LSTM(width),
            heads.ValueOutput(width))

    def forward(self, world, value=False):
        obs = world.obs
        outputs = arrdict.arrdict(
            logits=self.policy(world.obs, valid=world.valid))

        if value:
            #TODO: Maybe the env should handle this? 
            # Or there should be an output space for values? 
            v = self.value(obs, valid=world.valid)
            outputs['v'] = scatter_values(v, world.seats) if world.n_seats == 2 else v[..., None]
        return outputs
