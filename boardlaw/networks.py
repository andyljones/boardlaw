import numpy as np
import torch
from . import heads
from torch import nn
import torch.jit
from rebar import recurrence, arrdict, profiling
from torch.nn import functional as F
from collections import namedtuple

FIELDS = ('logits', 'v')

def positions(boardsize):
    # https://www.redblobgames.com/grids/hexagons/#conversions-axial
    #TODO: Does it help to sin/cos encode this?
    rs, cs = torch.meshgrid(
            torch.linspace(0, 1, boardsize),
            torch.linspace(0, 1, boardsize))
    zs = (rs + cs)/2.
    xs = torch.stack([rs, cs, zs], -1)

    # Since we're running over [0, 1] in 1/(b-1)-size steps, a period of 4/(b-1) 
    # gives a highest-freq pattern that goes
    #
    # sin:  0 +1  0 -1
    # cos  +1  0 -1  0
    #
    # every 4 steps. Then 16/(b-1) is just 4x slower than that, and between them
    # all the cells will be well-separated in dot product up to board size 13. 
    # 
    # Beyond that, the rightmost values in slower pattern'll start to get close 
    # to the leftmost values.
    periods = [4/(boardsize-1), 16/(boardsize-1)]
    if boardsize > 13:
        raise ValueError('Need to add support for boardsizes above 13')

    return torch.cat([torch.cat([torch.cos(2*np.pi*xs/p), torch.sin(2*np.pi*xs/p)], -1) for p in periods], -1)

class ReZeroConv(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, padding=1, stride=1, kernel_size=3, **kwargs)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        return x + self.α*F.relu(super().forward(x))

class GlobalContext(nn.Module):
    # Based on https://github.com/lucidrains/lightweight-gan/blob/main/lightweight_gan/lightweight_gan.py#L297

    def __init__(self, D):
        super().__init__()
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

        self.attn = nn.Conv2d(D, 1, 1, 1)
        self.compress = nn.Linear(D, D//2)
        self.expand = nn.Linear(D//2, D)

    def forward(self, x):
        attn = self.attn(x).flatten(2).softmax(dim=-1).squeeze(-2)
        vals = torch.einsum('bn,bcn->bc', attn, x.flatten(2))
        excited = self.expand(F.relu(self.compress(vals)))
        return x + self.α*excited[..., None, None]

class ConvContextModel(nn.Module):

    def __init__(self, obs_space, action_space, width=64, depth=8):
        super().__init__()
        boardsize = obs_space.dim[-2]

        layers = [nn.Conv2d(14, width, 3, 1, 1)]
        for l in range(depth):
            layers.append(ReZeroConv(width, width))
            layers.append(GlobalContext(width))
        self.layers = nn.ModuleList(layers)

        self.policy = nn.Conv2d(width, 1, 1)
        self.value1 = nn.Conv2d(width, 1, 1)
        self.value2 = nn.Linear(boardsize**2, 1)

        pos = positions(boardsize)
        self.register_buffer('pos', pos)

    @profiling.nvtx
    def traced(self, obs, valid, seats):
        if obs.ndim == 5:
            B, T = obs.shape[:2]
            p, v = self.traced(obs.flatten(0, 1), valid.flatten(0, 1), seats.flatten(0, 1))
            return p.reshape(B, T, -1), v.reshape(B, T, 2)

        B, boardsize, boardsize, _ = obs.shape
        prep = torch.cat([obs, self.pos[None].repeat_interleave(B, 0)], -1)
        x = prep.permute(0, 3, 1, 2)

        for l in self.layers:
            x = l(x)

        p = self.policy(x).flatten(1)
        p = p.where(valid, torch.full_like(p, -np.inf)).log_softmax(-1)

        v = F.relu(self.value1(x)).flatten(1)
        v = torch.tanh(self.value2(v).squeeze(-1))
        v = heads.scatter_values(v, seats)

        return p, v

    def forward(self, worlds):
        return arrdict.arrdict(zip(FIELDS, self.traced(worlds.obs, worlds.valid, worlds.seats)))

class ReZeroResidual(nn.Linear):

    def __init__(self, width):
        super().__init__(width, width)
        nn.init.orthogonal_(self.weight, gain=2**.5)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        return x + self.α*F.relu(super().forward(x))

class FCModel(nn.Module):

    def __init__(self, obs_space, action_space, width=128, depth=8):
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