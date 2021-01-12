from tqdm.auto import tqdm
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rebar import dotdict, arrdict

OFFSETS = [(-1, 0), (-1, 1), (0, -1), (0, 0), (0, +1), (-1, -1), (-1, 0)]

def plot(p):
    import matplotlib.pyplot as plt
    from boardlaw.hex import plot_board
    plot_board(np.stack(np.vectorize(plt.cm.RdBu)(.5+.5*p), -1))

class ReZeroResidual(nn.Linear):

    def __init__(self, width):
        super().__init__(width, width)
        nn.init.orthogonal_(self.weight, gain=2**.5)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        return x + self.α*F.relu(super().forward(x))

class FCModel(nn.Module):

    def __init__(self, Head, boardsize, D, n_layers=16):
        super().__init__()

        self.D = D
        layers = [nn.Linear(boardsize**2, D)]
        for _ in range(n_layers):
            layers.append(ReZeroResidual(D)) 
        self.layers = nn.ModuleList(layers)

        pos = positions(boardsize)
        self.register_buffer('pos', pos)

        self.head = Head(D, pos.shape[-1])

    def forward(self, obs):
        B, boardsize, boardsize, _ = obs.shape
        x = (obs[..., 0] - obs[..., 1]).reshape(B, boardsize*boardsize)
        for l in self.layers:
            x = F.relu(l(x))
        return self.head(x, self.pos)

class ReZeroConv(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, padding=1, stride=1, kernel_size=3, **kwargs)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        return x + self.α*F.relu(super().forward(x))

class ConvModel(nn.Module):

    def __init__(self, Head, boardsize, D, n_layers=16):
        super().__init__()

        layers = [nn.Conv2d(2, D, 3, 1, 1)]
        for l in range(n_layers):
            layers.append(ReZeroConv(D, D))
            
        layers.append(nn.Conv2d(D, 1, 3, 1, 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, obs):
        B, boardsize, boardsize, _ = obs.shape
        x = obs.permute(0, 3, 1, 2)
        for l in self.layers:
            x = l(x)
        x = x.reshape(B, -1)
        x = F.log_softmax(x, -1)
        return x.reshape(B, boardsize, boardsize)

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

    def __init__(self, Head, boardsize, D, n_layers=16):
        super().__init__()

        layers = [nn.Conv2d(14, D, 3, 1, 1)]
        for l in range(n_layers):
            layers.append(ReZeroConv(D, D))
            layers.append(GlobalContext(D))
            
        layers.append(nn.Conv2d(D, 1, 3, 1, 1))
        self.layers = nn.ModuleList(layers)

        pos = positions(boardsize)
        self.register_buffer('pos', pos)

    def forward(self, obs):
        B, boardsize, boardsize, _ = obs.shape
        prep = torch.cat([obs, self.pos[None].repeat_interleave(B, 0)], -1)
        x = prep.permute(0, 3, 1, 2)
        for l in self.layers:
            x = l(x)
        x = x.reshape(B, -1)
        x = F.log_softmax(x, -1)
        return x.reshape(B, boardsize, boardsize)

class FullAttention(nn.Module):
    # Based on https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/images.py

    def __init__(self, D):
        super().__init__()

        self.D = D

        self.qkv = nn.Conv2d(D, 3*D, 1, 1)
        self.out = nn.Conv2d(D, D, 3, 1, 1)

        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x):
        B, C, W, W = x.shape

        q, k, v = self.qkv(x).chunk(3, 1)

        q, k, v = map(lambda t: t.reshape(B, -1, W*W), (q, k, v))

        attn = torch.einsum('bcq,bck->bqk', q, k).div(self.D**.5).softmax(-1)
        out = torch.einsum('bqk,bck->bqc', attn, v).reshape(B, C, W, W)

        x = x + self.α*F.relu(out)
        x = x + self.α*F.relu(self.out(x))
        return x

class FullAttnModel(nn.Module):

    def __init__(self, Head, boardsize, D, n_layers=16):
        super().__init__()

        layers = [nn.Conv2d(14, D, 3, 1, 1)]
        for l in range(n_layers):
            layers.append(FullAttention(D))
            
        layers.append(nn.Conv2d(D, 1, 3, 1, 1))
        self.layers = nn.ModuleList(layers)

        pos = positions(boardsize)
        self.register_buffer('pos', pos)

    def forward(self, obs):
        B, boardsize, boardsize, _ = obs.shape
        prep = torch.cat([obs, self.pos[None].repeat_interleave(B, 0)], -1)
        x = prep.permute(0, 3, 1, 2)
        for l in self.layers:
            x = l(x)
        x = x.reshape(B, -1)
        x = F.log_softmax(x, -1)
        return x.reshape(B, boardsize, boardsize)


class HybridModel(nn.Module):

    def __init__(self, Head, boardsize, D, n_layers=16):
        super().__init__()

        layers = [
            nn.Conv2d(2, D, 3, 1, 1),
            ReZeroConv(D, D)]

            
        layers.append(nn.Conv2d(D, 1, 3, 1, 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, obs):
        B, boardsize, boardsize, _ = obs.shape
        x = obs.permute(0, 3, 1, 2)
        for l in self.layers:
            x = l(x)
        x = x.reshape(B, -1)
        x = F.log_softmax(x, -1)
        return x.reshape(B, boardsize, boardsize)




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

        return F.relu(self.final(vals.view(B, H*Dx))), attn.detach()

class ReZeroAttn(nn.Module):

    def __init__(self, D, *args, **kwargs):
        super().__init__()
        self.attn = Attention(D, *args, **kwargs)
        self.fc0 = nn.Linear(D, D)
        self.fc1 = nn.Linear(D, D)

        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, b):
        y, a = self.attn(x, b)
        x = x + self.α*y

        y = F.relu(self.fc0(x))
        y = self.fc1(y)
        x = x + self.α*y

        return x, a

class AttnModel(nn.Module):

    def __init__(self, Head, boardsize, D, n_layers=8, n_heads=1):
        super().__init__()

        pos = positions(boardsize)
        self.register_buffer('pos', pos)

        exemplar = torch.zeros((1, boardsize, boardsize, 2))
        D_prep = prepare(exemplar, pos).shape[-1]

        self.D = D
        layers = []
        for _ in range(n_layers):
            layers.append(ReZeroAttn(D, D_prep, H=n_heads)) 
        self.layers = nn.ModuleList(layers)

        self.head = Head(D, pos.shape[-1])

    def forward(self, obs):
        b = prepare(obs, self.pos)
        x = torch.zeros((obs.shape[0], self.D), device=obs.device)
        attns = []
        for l in self.layers:
            x, a = l(x, b)
            attns.append(a)
        logits = self.head(x, self.pos)

        n_heads = attns[0].shape[-1]
        attns.append(logits.exp().reshape(logits.shape[0], -1, 1).repeat_interleave(n_heads, -1))
        self.attns = torch.stack(attns, 1)
        return logits
