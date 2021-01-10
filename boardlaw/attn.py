from tqdm.auto import tqdm
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rebar import dotdict, arrdict
from . import heads

OFFSETS = [(-1, 0), (-1, 1), (0, -1), (0, 0), (0, +1), (-1, -1), (-1, 0)]

def positions(boardsize):
    # https://www.redblobgames.com/grids/hexagons/#conversions-axial
    #TODO: Does it help to sin/cos encode this?
    rs, cs = torch.meshgrid(
            torch.linspace(-1, 1, boardsize),
            torch.linspace(-1, 1, boardsize))
    zs = (rs + cs)/2.
    xs = torch.stack([rs, cs, zs], -1)

    ps = [1, 2, 4]
    return torch.cat([
        torch.cat([torch.cos(2*np.pi*xs/p) for p in ps], -1),
        torch.cat([torch.sin(2*np.pi*xs/p) for p in ps], -1)], -1)

def offset(board, o):
    w = board.shape[-1]
    r, c = o
    t, b = 1+r, w-1+r
    l, r = 1+c, w-1+c
    return board[..., t:b, l:r]

def neighbourhoods(obs):
    single = obs[..., 0] - obs[..., 1]
    augmented = F.pad(single, (1, 1, 1, 1))
    return torch.stack([offset(augmented, o) for o in OFFSETS], -1)

def prepare(obs, pos):
    pos = pos[None].repeat_interleave(obs.shape[0], 0)
    stack = torch.cat([neighbourhoods(obs), pos], -1)
    B, H, W, C = stack.shape
    return stack.reshape(B, H*W, C)

class PosActions(nn.Module):

    def __init__(self, D, D_pos):
        super().__init__()
        self.k_p = nn.Linear(D_pos, D) 
        self.k_x = nn.Linear(D, D) 
        self.q = nn.Linear(D, D)

    def forward(self, x, p):
        B, D = x.shape
        boardsize = p.size(-2)

        p = p.view(boardsize*boardsize, -1)
        k = self.k_x(x)[:, None, :] + self.k_p(p)[None, :, :]
        q = self.q(x)

        dots = torch.einsum('bpd,bd->bp', k, q)/D**.5

        return F.log_softmax(dots, -1).reshape(B, boardsize, boardsize)

class Attention(nn.Module):

    def __init__(self, D, D_prep, H=1):
        super().__init__()

        self.H = H
        self.kv_x = nn.Linear(D, 2*H*D)
        self.kv_b = nn.Linear(D_prep, 2*H*D)
        self.q = nn.Linear(D, D*H)

        self.final = nn.Linear(D*H, D)

    def forward(self, x, b):
        B, Dx = x.shape
        B, P, Db = b.shape
        H = self.H

        k, v = (self.kv_x(x)[:, None, :] + self.kv_b(b)).chunk(2, -1)
        q = self.q(x)

        k = k.view(B, P, H, Dx)
        v = v.view(B, P, H, Dx)
        q = q.view(B, H, Dx)

        dots = torch.einsum('bphd,bhd->bph', k, q)/Dx**.5
        attn = torch.softmax(dots, -2)
        vals = torch.einsum('bph,bphd->bhd', attn, v)

        return F.relu(self.final(vals.view(B, H*Dx)))

class ReZeroAttn(nn.Module):

    def __init__(self, D, *args, **kwargs):
        super().__init__()
        self.attn = Attention(D, *args, **kwargs)
        self.fc0 = nn.Linear(D, D)
        self.fc1 = nn.Linear(D, D)

        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, b):
        y = self.attn(x, b)
        x = x + self.α*y

        y = F.relu(self.fc0(x))
        y = self.fc1(y)
        x = x + self.α*y

        return x

FIELDS = ('logits', 'v')

class Network(nn.Module):

    def __init__(self, obs_space, action_space, D=32, n_layers=8):
        super().__init__()
        self.D = D
        self.policy = heads.output(action_space, D)
        self.sampler = self.policy.sample

        boardsize = obs_space.dim[-2]
        pos = positions(boardsize).cuda()
        self.register_buffer('pos', pos)

        exemplar = torch.zeros((1, boardsize, boardsize, 2)).cuda()
        D_prep = prepare(exemplar, pos).shape[-1]

        blocks = []
        for _ in range(n_layers):
            blocks.append(ReZeroAttn(D, D_prep))
        self.body = nn.ModuleList(blocks) 

        self.value = heads.ValueOutput(D)

    def traced(self, obs, valid, seats):
        b = prepare(obs, self.pos)
        x = torch.zeros((obs.shape[0], self.D), device=obs.device)
        for l in self.body:
            x = l(x, b)
        return (self.policy(x, valid), self.value(x, valid, seats))

    def forward(self, worlds):
        return arrdict.arrdict(zip(FIELDS, self.traced(worlds.obs, worlds.valid, worlds.seats)))

