import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rebar import dotdict

OFFSETS = [(-1, 0), (-1, 1), (0, -1), (0, 0), (0, +1), (-1, -1), (-1, 0)]

def plot(p):
    import matplotlib.pyplot as plt
    from .hex import plot_board
    plot_board(np.stack(np.vectorize(plt.cm.RdBu)(.5+.5*p), -1))

class Mechanical(nn.Module):

    def __init__(self, boardsize, D):
        super().__init__()

    def forward(self, obs):
        x = 10*(obs.sum(-1) - 1) - 1
        x = x.reshape(obs.shape[0], -1)
        x = F.log_softmax(x, -1)
        x = x.reshape(obs.shape[:-1])
        return x

class ReZeroResidual(nn.Linear):

    def __init__(self, width):
        super().__init__(width, width)
        nn.init.orthogonal_(self.weight, gain=2**.5)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        return x + self.α*F.relu(super().forward(x))

class FCModel(nn.Module):

    def __init__(self, boardsize, D, n_layers=8):
        super().__init__()

        self.D = D
        self.first = nn.Linear(boardsize**2, D)
        # self.body = nn.Sequential(*[ReZeroResidual(D) for _ in range(n_layers)])
        self.second = nn.Linear(D, boardsize**2)

        pos = positions(boardsize)
        self.register_buffer('pos', pos)

    def forward(self, obs):
        B, boardsize, boardsize, _ = obs.shape
        x = (obs[..., 0] - obs[..., 1]).reshape(B, boardsize*boardsize)
        x = F.relu(self.first(x))
        x = self.second(x)
        x = F.log_softmax(x, -1)
        return x.reshape(B, boardsize, boardsize)

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

    def __init__(self, boardsize, D, D_pos):
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

    def __init__(self, D, D_obs, D_pos, H=1):
        super().__init__()

        self.H = H
        self.kv_x = nn.Linear(D, 2*H*D)
        self.kv_b = nn.Linear(D_obs+D_pos, 2*H*D)
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

        out = self.final(vals.view(B, H*Dx))

        return out

class Model(nn.Module):

    def __init__(self, boardsize, D, D_obs=7):
        super().__init__()

        pos = positions(boardsize)
        self.register_buffer('pos', pos)
        D_pos = pos.size(-1)

        self.D = D
        self.first = nn.Linear(D, D)
        self.attn = Attention(D, D_obs, D_pos, H=2)
        self.second = nn.Linear(D, D)
        self.policy0 = PosActions(boardsize, D, D_pos)
        self.policy1 = PosActions(boardsize, D, D_pos)

    def forward(self, obs):
        b = prepare(obs, self.pos)
        x = torch.zeros((obs.shape[0], self.D), device=obs.device)
        x = F.relu(self.first(x))
        x = self.attn(x, b)
        x = self.second(x)
        x0 = self.policy0(x, self.pos)
        x1 = self.policy1(x, self.pos)
        return x0, x1

def pointer_loss(rows, cols, boardsize, outputs):
    targets = 2*torch.stack([rows/boardsize, cols/boardsize], -1) - 1
    return (targets - outputs).pow(2).mean()

def action_loss(rows, cols, boardsize, outputs):
    B = rows.shape[0]
    targets = rows*boardsize + cols
    return F.nll_loss(outputs.reshape(B, -1), targets)

def test():
    from boardlaw.hex import Hex
    from tqdm.auto import tqdm

    worlds = Hex.initial(1, 5)

    T = 5000
    D = 32

    model = Model(worlds.boardsize, D).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    B = 8*1024
    envs = torch.arange(B, device=worlds.device)

    losses = []
    with tqdm(total=T) as pbar:
        for t in range(T):
            obs = torch.zeros((B, worlds.boardsize, worlds.boardsize, 2), device=worlds.device)

            r0 = torch.randint(0, worlds.boardsize, size=(B,), device=worlds.device)
            c0 = torch.randint(0, worlds.boardsize, size=(B,), device=worlds.device)
            obs[envs, r0, c0, 0] = 1.

            r1 = torch.randint(0, worlds.boardsize, size=(B,), device=worlds.device)
            c1 = torch.randint(0, worlds.boardsize, size=(B,), device=worlds.device)
            obs[envs, r1, c1, 1] = 1.
            
            x0, x1 = model(obs)
            loss = action_loss(r0, c0, worlds.boardsize, x0) + action_loss(r1, c1, worlds.boardsize, x1)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_description(f'{loss:.2f}')
            if t % 10 == 0:
                pbar.update(10)

            losses.append(float(loss))

    return dotdict.dotdict(
        losses=pd.Series(losses),
        obs=obs,
        outputs=outputs,
        targets=targets)