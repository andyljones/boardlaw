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
        super().__init__(*args, padding=0, stride=1, kernel_size=3, **kwargs)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        y = self.α*F.relu(super().forward(x))
        y[:, :self.in_channels] += x[:, :, 1:-1, 1:-1]
        return y

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

    def __init__(self, obs_space, action_space, width=32, depth=8):
        super().__init__()
        boardsize = obs_space.dim[-2]

        self.convs = nn.ModuleList([
            ReZeroConv(2, 8),
            ReZeroConv(8, 16),
            ReZeroConv(16, 32)])
        self.body = nn.ModuleList([
            ReZeroResidual(32*3*3, 128),
            ReZeroResidual(128, 128),
            ReZeroResidual(128, 128),
            ReZeroResidual(128, 128),
            ReZeroResidual(128, 128)])

        self.policy = heads.MaskedOutput(action_space, 128)
        self.value = heads.ValueOutput(128)

        pos = positions(boardsize)
        self.register_buffer('pos', pos)

    @profiling.nvtx
    def traced(self, obs, valid, seats):
        if obs.ndim == 5:
            B, T = obs.shape[:2]
            p, v = self.traced(obs.flatten(0, 1), valid.flatten(0, 1), seats.flatten(0, 1))
            return p.reshape(B, T, -1), v.reshape(B, T, 2)

        B, boardsize, boardsize, _ = obs.shape
        x = obs.permute(0, 3, 1, 2)

        for l in self.convs:
            x = l(x)
        x = x.flatten(1)
        for l in self.body:
            x = l(x)

        p = self.policy(x, valid)
        v = self.value(x, valid, seats)

        return p, v

    def forward(self, worlds):
        return arrdict.arrdict(zip(FIELDS, self.traced(worlds.obs, worlds.valid, worlds.seats)))

class ReZeroResidual(nn.Linear):

    def __init__(self, i, o):
        super().__init__(i, o)
        nn.init.orthogonal_(self.weight, gain=2**.5)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        y = self.α*F.relu(super().forward(x))
        y = x[:, :self.out_features]
        return y

class Network(nn.Module):

    def __init__(self, obs_space, action_space, width=256, depth=8):
        super().__init__()
        self.policy = heads.output(action_space, width)
        self.sampler = self.policy.sample

        blocks = [heads.intake(obs_space, width)]
        for _ in range(depth):
            blocks.append(ReZeroResidual(width))
        self.body = nn.Sequential(*blocks) 

        self.value = heads.ValueOutput(width)

    def traced(self, obs, valid, seats):
        neck = self.body(obs)
        return (self.policy(neck, valid), self.value(neck, valid, seats))

    def forward(self, worlds):
        return arrdict.arrdict(zip(FIELDS, self.traced(worlds.obs, worlds.valid, worlds.seats)))

def traced_network(obs_space, action_space, *args, **kwargs):
    #TODO: This trace all has to be done with standins of the right device,
    # else the full_likes I've got scattered through my code will break.
    n = Network(obs_space, action_space, *args, **kwargs).cuda()
    obs = torch.ones((4, *obs_space.dim), dtype=torch.float, device='cuda')
    valid = torch.ones((4, action_space.dim), dtype=torch.bool, device='cuda')
    seats = torch.ones((4,), dtype=torch.int, device='cuda')
    m = torch.jit.trace_module(n, {'traced': (obs, valid, seats)})
    m.forward = n.forward
    return m

class LeagueNetwork(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        # self.prime = traced_network(*args, **kwargs)
        self.prime = ConvContextModel(*args, **kwargs)
        self.opponents = nn.ModuleList()
        self.slices = []

        self.streams = [torch.cuda.Stream() for _ in range(2)]

        self.prime_only = True
    
    @profiling.nvtx
    def forward_one(self, net, obs, valid, seats):
        return dict(zip(FIELDS, net.traced(obs, valid, seats)))

    @profiling.nvtx
    def forward(self, worlds):
        if self.prime_only:
            split = worlds.n_envs
        else:
            split = min([s.start for s in self.slices], default=worlds.n_envs)

        parts = []
        obs, valid, seats = worlds.obs, worlds.valid, worlds.seats

        torch.cuda.synchronize()
        with torch.cuda.stream(self.streams[0]):
            s = slice(0, split)
            parts.append(self.forward_one(self.prime, obs[s], valid[s], seats[s]))

        if split < worlds.n_envs:
            chunk = (worlds.n_envs - split)//len(self.opponents)
            assert split + chunk*len(self.opponents) == worlds.n_envs
            with torch.cuda.stream(self.streams[1]):
                for s, opponent in zip(self.slices, self.opponents): 
                    parts.append(self.forward_one(opponent, obs[s], valid[s], seats[s]))

        torch.cuda.synchronize()
        return arrdict.from_dicts(arrdict.cat(parts))

    def state_dict(self):
        return self.prime.state_dict()

    def load_state_dict(self, sd):
        self.prime.load_state_dict(sd)
        for oppo in self.opponents:
            oppo.load_state_dict(sd)

class SimpleNetwork(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.prime = traced_network(*args, **kwargs)

    def forward(self, worlds):
        resp = arrdict.from_dicts(dict(zip(FIELDS, self.prime.traced(worlds.obs, worlds.valid, worlds.seats))))
        return resp

    def state_dict(self):
        return self.prime.state_dict()

    def load_state_dict(self, sd):
        self.prime.load_state_dict(sd)