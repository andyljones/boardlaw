import numpy as np
import torch
from . import heads, lstm
from torch import nn
from rebar import recurrence, arrdict
from torch.nn import functional as F

class LayerNorm2D(nn.LayerNorm):

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class Residual(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = LayerNorm2D(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = LayerNorm2D(channels)

    def forward(self, x, *args, **kwargs):
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return F.relu(x + y)

class Flatten(nn.Module):

    def forward(self, x):
        return x.reshape(*x.shape[:-3], -1)

class Network(nn.Module):

    def __init__(self, obs_space, action_space, width=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, width, 1),
            Residual(width),
            Residual(width),
            Flatten())

        example_in = torch.zeros((1, *obs_space.dim))
        example_out = self.body(example_in.permute(0, 3, 1, 2))
        n_out = example_out.nelement()
        
        self.neck = nn.Sequential(
            nn.Linear(n_out, 4*width),
            nn.ReLU())
        self.policy = heads.output(action_space, 4*width)
        self.value = heads.ValueOutput(4*width)

    # def trace(self, world):
    #     self.policy = torch.jit.trace_module(self.policy, {'forward': (world.obs, world.valid)})
    #     self.vaue = torch.jit.trace_module(self.value, {'forward': (world.obs, world.valid, world.seats)})

    def forward(self, world, value=False):
        b = self.body(world.obs.permute(0, 3, 1, 2))
        n = self.neck(b)
        outputs = arrdict.arrdict(
            logits=self.policy(n, valid=world.valid))

        if value:
            #TODO: Maybe the env should handle this? 
            # Or there should be an output space for values? 
            outputs['v'] = self.value(n, valid=world.valid, seats=world.seats)
        return outputs
