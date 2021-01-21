import numpy as np
import torch
from . import heads
from torch import nn
import torch.jit
from rebar import recurrence, arrdict, profiling
from torch.nn import functional as F
from collections import namedtuple

FIELDS = ('logits', 'v')

class ReZeroResidual(nn.Linear):

    def __init__(self, width):
        super().__init__(width, width)
        nn.init.orthogonal_(self.weight, gain=2**.5)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        return x + self.α*F.relu(super().forward(x))

class FCModel(nn.Module):

    def __init__(self, obs_space, action_space, width=256, depth=20):
        super().__init__()
        self.policy = heads.output(action_space, width)
        self.sampler = self.policy.sample

        blocks = [heads.intake(obs_space, width)]
        for _ in range(depth):
            blocks.append(ReZeroResidual(width))
        self.body = nn.Sequential(*blocks) 

        self.value = heads.ValueOutput(width)

    def forward(self, worlds):
        neck = self.body(worlds.obs)
        return arrdict.arrdict(
            logits=self.policy(neck, worlds.valid), 
            v=self.value(neck, worlds.valid, worlds.seats))

class SplitLayer(nn.Module):

    def __init__(self, width):
        super().__init__()

        self.P = nn.Linear(width, width//2)
        nn.init.orthogonal_(self.P.weight, gain=2**.5)

        self.V = nn.Linear(width, width//2)
        nn.init.orthogonal_(self.V.weight, gain=2**.5)

        self.register_parameter('α', nn.Parameter(torch.zeros((4,))))

    def forward(self, x, *args, **kwargs):
        p, v = x.chunk(2, -1)
        
        αpp, αpv, αvv, αvp = self.α

        p = p + F.relu(self.P(torch.cat([αpp*p, αvp*v.detach()], -1)))
        v = v + F.relu(self.V(torch.cat([αpv*p.detach(), αvv*v], -1)))

        return torch.cat([p, v], -1)

class SplitModel(nn.Module):

    def __init__(self, obs_space, action_space, width=256, depth=20):
        super().__init__()
        self.policy = heads.output(action_space, width)
        self.sampler = self.policy.sample

        blocks = [heads.intake(obs_space, width)]
        for _ in range(depth):
            blocks.append(SplitLayer(width))
        self.body = nn.Sequential(*blocks) 

        self.value = heads.ValueOutput(width)

    def forward(self, worlds):
        neck = self.body(worlds.obs)
        return arrdict.arrdict(
            logits=self.policy(neck, worlds.valid), 
            v=self.value(neck, worlds.valid, worlds.seats))