import numpy as np
import torch
from . import heads
from torch import nn
from rebar import recurrence, arrdict
from torch.nn import functional as F

class Residual(nn.Module):

    def __init__(self, width, gain=1):
        # "Identity Mappings in Deep Residual Networks"
        super().__init__()
        self.w0 = nn.Linear(width, width, bias=False)
        self.n0 = nn.LayerNorm(width)
        self.w1 = nn.Linear(width, width, bias=False)
        self.n1 = nn.LayerNorm(width)

        nn.init.orthogonal_(self.w0.weight)
        nn.init.orthogonal_(self.w1.weight, gain=gain)

    def forward(self, x, *args, **kwargs):
        y = self.n0(x)
        y = F.relu(y)
        y = self.w0(y)
        y = self.n1(y)
        y = F.relu(y)
        y = self.w1(y)
        return x + y

class Transformer(nn.Module):

    def __init__(self, boardsize, width):
        super().__init__()
        self.boardsize = boardsize
        self.width = width
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.full = nn.Linear(width, width)

    def forward(self, x, *args, **kwargs):
        T, B, H, W, C = x.shape
        n = self.norm(x.reshape(-1, self.boardsize**2, self.width))
        q, k, v = self.qkv(n).chunk(3, -1)

        sim = torch.einsum('bqc,bkc->bqkc', q, k)
        attn = torch.softmax(sim, 2)

        attended = torch.einsum('bqkc,bkc->bqc', attn, v)
        attended = attended.reshape(T, B, H, W, C)
        
        y = F.relu(self.full(attended))

        return x + y

class Network(nn.Module):

    def __init__(self, obs_space, action_space, width=32, neckwidth=4, layers=4):
        super().__init__()
        boardsize = obs_space.dim[0]
        self.policy = heads.output(action_space, neckwidth*boardsize**2)
        self.sampler = self.policy.sample

        blocks = [nn.Linear(obs_space.dim[-1], width)]
        for _ in range(layers):
            blocks.append(Transformer(boardsize, width)) 
        blocks.append(nn.Linear(width, 4))
        self.body = nn.Sequential(*blocks)

        self.value = heads.ValueOutput(neckwidth*boardsize**2)

    # def trace(self, world):
    #     self.policy = torch.jit.trace_module(self.policy, {'forward': (world.obs, world.valid)})
    #     self.vaue = torch.jit.trace_module(self.value, {'forward': (world.obs, world.valid, world.seats)})

    def forward(self, world, value=False):
        if world.obs.ndim == 4:
            return self.forward(world[None], value).squeeze(0)

        T, B, H, W, C = world.obs.shape
        neck = self.body(world.obs).reshape(T, B, -1)
        outputs = arrdict.arrdict(
            logits=self.policy(neck, valid=world.valid))

        if value:
            #TODO: Maybe the env should handle this? 
            # Or there should be an output space for values? 
            outputs['v'] = self.value(neck, valid=world.valid, seats=world.seats)
        return outputs

def check_var():
    from boardlaw.main.common import worldfunc
    import pandas as pd

    worlds = worldfunc(256)
    stds = {}
    for n in range(1, 20):
        net = Network(worlds.obs_space, worlds.action_space, layers=n).cuda()

        obs = torch.rand_like(worlds.obs)
        obs.requires_grad = True            
        
        l = net.body(obs)
        sf = l.std().item()
        
        for p in net.parameters():
            p.grad = None
        
        l.sum().backward()
        
        stds[n] = {'forward': sf, 'backward': obs.grad.std().item()}
    stds = pd.DataFrame(stds).T

def test_transformer():

    boardsize = 3
    width = 4

    model = Transformer(boardsize, width)

    obs = torch.zeros((5, 7, boardsize, boardsize, width))
    out = model(obs)

    from boardlaw.main.common import worldfunc
    worlds = worldfunc(32)

    model = Network(worlds.obs_space, worlds.action_space).cuda()
    model(worlds)