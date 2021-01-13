import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rebar import arrdict

FIELDS = ('logits', 'v')

def scatter_values(v, seats):
    seats = torch.stack([seats, 1-seats], -1)
    vs = torch.stack([v, -v], -1)
    xs = torch.full_like(vs, np.nan)
    return xs.scatter(-1, seats.long(), vs)

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

    def __init__(self, boardsize, width=16, depth=8):
        super().__init__()

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
        v = scatter_values(v, seats)

        return p, v

    def forward(self, worlds):
        return arrdict.arrdict(zip(FIELDS, self.traced(worlds.obs, worlds.valid, worlds.seats)))

def run():
    buffer_length = 16 
    batch_size = 64*1024
    n_envs = 8*1024
    boardsize = 7
    buffer_inc = batch_size//n_envs
    device = 'cuda'

    network = ConvContextModel(boardsize).to(device)
    opt = torch.optim.Adam(network.parameters(), lr=1e-2, amsgrad=True)
    scaler = torch.cuda.amp.GradScaler()


    while True:
        w = arrdict.arrdict(
            obs=torch.zeros((batch_size, boardsize, boardsize, 2), device=device),
            seats=torch.randint(0, 2, (batch_size,), device=device),
            valid=torch.randint(0, 2, (batch_size, boardsize**2), device=device).bool())
        d0 = arrdict.arrdict(
            logits=-torch.ones((batch_size, boardsize**2), device=device),
            v=torch.zeros((batch_size, 2), device=device))
        
        reward_to_go = torch.zeros((batch_size, 2), device=device)
        
        mask = torch.randint(0, 2, (batch_size,), device=device).bool()

        with torch.cuda.amp.autocast():
            d = network(w)

            zeros = torch.zeros_like(d.logits)
            policy_loss = -(d0.logits.float().exp()*d.logits).where(w.valid, zeros).sum(axis=-1)[mask].mean()

            target_value = reward_to_go
            value_loss = (target_value - d.v).square()[mask].mean()

            loss = policy_loss + value_loss 

        old = torch.cat([p.flatten() for p in network.parameters()])

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        print('Stepped')

if __name__ == '__main__':
    run()