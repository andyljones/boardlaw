import numpy as np
import torch
from . import heads
from torch import nn
import torch.jit
from rebar import recurrence, arrdict, profiling
from torch.nn import functional as F
from collections import namedtuple

class ReZeroResidual(nn.Linear):

    def __init__(self, width):
        super().__init__(width, width)
        nn.init.orthogonal_(self.weight, gain=2**.5)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        return x + self.α*super().forward(F.relu(x))

class FCModel(nn.Module):

    def __init__(self, obs_space, action_space, width=256, depth=64):
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space

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