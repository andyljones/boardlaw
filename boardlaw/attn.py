"""
Idea:
    * Attend to the board
    * Say main layers are 256 neurons
    * Then at each layer, compress the input down to 8
    * At each location, stack it with the position (4?) and 3x3 board features (9 or 18, depending)
    * That gives 8 + 4 + 18 = 30 input, which you can stack with the 8 and transform into 16?
    * On a 9x9 board, that works out to about the cost of one 256x256 layer
    * Attend to it with a 16 key, return a 16 value, expand that value up to 256 again and add it in
    * Then ReZero it into the trunk.

"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OFFSETS = [(-1, 0), (-1, 1), (0, -1), (0, 0), (0, +1), (-1, -1), (-1, 0)]

def plot(p):
    import matplotlib.pyplot as plt
    from .hex import plot_board
    plot_board(np.stack(np.vectorize(plt.cm.RdBu)(.5+.5*p), -1))

def positions(boardsize):
    # https://www.redblobgames.com/grids/hexagons/#conversions-axial
    #TODO: Does it help to sin/cos encode this?
    rs, cs = torch.meshgrid(
            torch.linspace(-1, 1, boardsize),
            torch.linspace(-1, 1, boardsize))
    zs = (rs + cs)/2.
    return torch.stack([rs, cs, zs], -1)

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

class Attention(nn.Module):

    def __init__(self, D=16, H=1):
        super().__init__()

        self.H = H
        D_obs, D_pos = 7, 3
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
        attn = torch.softmax(dots, -1)
        vals = torch.einsum('bph,bphd->bhd', attn, v)

        out = self.final(vals.view(B, H*Dx))

        return out

class MaskedActions(nn.Module):

    def __init__(self, boardsize, D):
        super().__init__()
        D_pos = 3
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

class Model(nn.Module):

    def __init__(self, boardsize, D):
        super().__init__()

        self.D = D
        self.first = nn.Linear(D, D)
        self.attn = Attention(D)
        self.second = nn.Linear(D, D)
        self.policy = MaskedActions(boardsize, D)

        pos = positions(boardsize)
        self.register_buffer('pos', pos)

    def forward(self, obs):
        b = prepare(obs, self.pos)
        x = torch.zeros((obs.shape[0], self.D), device=obs.device)
        x = F.relu(self.first(x))
        x = self.attn(x, b)
        x = F.relu(self.second(x))
        x = self.policy(x, self.pos)
        return x

def test():
    from boardlaw.hex import Hex
    from tqdm.auto import tqdm

    worlds = Hex.initial(1, 5)

    D = 16

    model = Model(worlds.boardsize, D).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    B = 1024
    envs = torch.arange(B, device=worlds.device)
    with tqdm() as pbar:
        while True:
            rows = torch.randint(0, worlds.boardsize, size=(B,), device=worlds.device)
            cols = torch.randint(0, worlds.boardsize, size=(B,), device=worlds.device)

            obs = torch.zeros((B, worlds.boardsize, worlds.boardsize, 2), device=worlds.device)
            obs[envs, rows, cols, 0] = 1.
            targets = rows*worlds.boardsize + cols

            logprobs = model(obs)

            loss = F.nll_loss(logprobs.reshape(B, -1), targets)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_description(f'{loss:.2f}')
