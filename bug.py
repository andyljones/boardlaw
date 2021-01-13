import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import torch
from torch import nn

class Model(nn.Module):

    def __init__(self, channels=16):
        super().__init__()

        layers = [nn.Conv2d(2, channels, 3, 1, 1)]
        for l in range(16):
            layers.append(nn.Conv2d(channels, channels, 3, 1, 1))
        layers.append(nn.Conv2d(channels, 1, 1))

        self.layers = nn.ModuleList(layers)

    def forward(self, obs):
        x = obs.permute(0, 3, 1, 2)

        for l in self.layers:
            x = l(x)

        return x.flatten(1).log_softmax(-1)

def run():
    batch = 64*1024
    width = 7
    device = 'cuda'

    network = Model().to(device)
    opt = torch.optim.Adam(network.parameters(), amsgrad=True)
    scaler = torch.cuda.amp.GradScaler()

    obs = torch.zeros((batch, width, width, 2), device=device)
    yhat = -torch.ones((batch, width**2), device=device)
    
    mask = torch.ones((batch,), device=device).bool()

    while True:
        with torch.cuda.amp.autocast():
            y = network(obs)
            loss = -(yhat.exp()*y).sum(axis=-1)[mask].mean()

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        print('Stepped')

if __name__ == '__main__':
    run()