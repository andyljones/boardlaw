from torch import nn
import torch
from torch.nn import functional as F
from boardlaw.heads import scatter_values

class ReZeroResidual(nn.Linear):

    def __init__(self, width):
        super().__init__(width, width)
        nn.init.orthogonal_(self.weight, gain=2**.5)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        return x + self.α*F.relu(super().forward(x))

class FCModel(nn.Module):

    def __init__(self, boardsize, width=256, depth=20):
        super().__init__()

        blocks = [nn.Linear(2*boardsize**2, width)]
        for _ in range(depth):
            blocks.append(ReZeroResidual(width))
        self.body = nn.Sequential(*blocks) 

        self.value = nn.Linear(width, 1)

    def forward(self, obs, seats):
        obs = obs.flatten(1)
        neck = self.body(obs)
        v = self.value(neck).squeeze(-1)
        return scatter_values(torch.tanh(v), seats)

class ReZeroConv(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, padding=1, stride=1, kernel_size=3, **kwargs)
        self.register_parameter('α', nn.Parameter(torch.zeros(())))

    def forward(self, x, *args, **kwargs):
        return x + self.α*F.relu(super().forward(x))

class ConvModel(nn.Module):

    def __init__(self, boardsize, D, n_layers=16):
        super().__init__()

        layers = [nn.Conv2d(2, D, 3, 1, 1)]
        for l in range(n_layers):
            layers.append(ReZeroConv(D, D))
        layers.append(nn.Conv2d(D, 1, 3, 1, 1))
        self.layers = nn.ModuleList(layers)

        self.value = nn.Linear(boardsize**2, 1)

    def forward(self, obs, seats):
        B, boardsize, boardsize, _ = obs.shape
        x = obs.permute(0, 3, 1, 2)
        for l in self.layers:
            x = l(x)
        x = x.reshape(B, -1)
        v = self.value(x.flatten(1)).squeeze(-1)
        return scatter_values(torch.tanh(v), seats)

