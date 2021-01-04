import numpy as np
import torch
from . import heads
from torch import nn
import torch.jit
from rebar import recurrence, arrdict
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

    def forward(self, obs, valid, seats):
        neck = self.body(obs)
        return (self.policy(neck, valid), self.value(neck, valid, seats))

def traced_network(obs_space, action_space, *args, **kwargs):
    #TODO: This trace all has to be done with standins of the right device,
    # else the full_likes I've got scattered through my code will break.
    n = Network(obs_space, action_space, *args, **kwargs).cuda()
    obs = torch.ones((4, *obs_space.dim), dtype=torch.float, device='cuda')
    valid = torch.ones((4, action_space.dim), dtype=torch.bool, device='cuda')
    seats = torch.ones((4,), dtype=torch.int, device='cuda')
    return torch.jit.trace(n, (obs, valid, seats))

class LeagueNetwork(nn.Module):

    def __init__(self, *args, n_opponents=4, split=.75, **kwargs):
        super().__init__()

        self.latest = traced_network(*args, **kwargs)
        self.split = split
        self.opponents = nn.ModuleList([traced_network(*args, **kwargs) for _ in range(n_opponents)])
        self.n_opponents = n_opponents

        self.streams = [torch.cuda.Stream() for _ in range(2)]

    def forward(self, worlds):
        torch.cuda.synchronize()
        split = int(self.split*worlds.n_envs) if self.n_opponents else worlds.n_envs

        parts = []
        obs, valid, seats = worlds.obs, worlds.valid, worlds.seats
        with torch.cuda.stream(self.streams[0]):
            s = slice(0, split)
            part = dict(zip(FIELDS, self.latest(obs[s], valid[s], seats[s])))
            part['agentid'] = torch.full((split,), True, device=worlds.device)
            parts.append(part)

        if self.n_opponents:
            chunk = (worlds.n_envs - split)//len(self.opponents)
            assert split + chunk*len(self.opponents) == worlds.n_envs
            with torch.cuda.stream(self.streams[1]):
                for i, opponent in enumerate(self.opponents):
                    s = slice(split + i*chunk, split + (i+1)*chunk)
                    part = dict(zip(FIELDS, opponent(obs[s], valid[s], seats[s])))
                    part['agentid'] = torch.full((split,), False, device=worlds.device)
                    parts.append(part)

        torch.cuda.synchronize()
        return arrdict.from_dicts(arrdict.cat(parts))
        