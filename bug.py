import numpy as np
import torch
from torch import nn

class Model(nn.Module):

    def __init__(self, width=16, depth=16):
        super().__init__()

        layers = [nn.Conv2d(14, width, 3, 1, 1)]
        for l in range(depth):
            layers.append(nn.Conv2d(width, width, 3, 1, 1))
        self.layers = nn.ModuleList(layers)
        self.policy = nn.Conv2d(width, 1, 1)

    def forward(self, obs, valid):
        B, boardsize, boardsize, _ = obs.shape
        prep = torch.cat([obs, 
            torch.zeros((B, boardsize, boardsize, 12), device=obs.device)], -1)
        x = prep.permute(0, 3, 1, 2)

        for l in self.layers:
            x = l(x)

        p = self.policy(x).flatten(1)
        p = p.where(valid, torch.full_like(p, -np.inf)).log_softmax(-1)

        return p

def run():
    batch_size = 64*1024
    boardsize = 7
    device = 'cuda'

    network = Model().to(device)
    opt = torch.optim.Adam(network.parameters(), lr=1e-2, amsgrad=True)
    scaler = torch.cuda.amp.GradScaler()

    while True:
        obs=torch.zeros((batch_size, boardsize, boardsize, 2), device=device)
        valid=torch.randint(0, 2, (batch_size, boardsize**2), device=device).bool()
        logits0 = -torch.ones((batch_size, boardsize**2), device=device)
        
        mask = torch.randint(0, 2, (batch_size,), device=device).bool()

        with torch.cuda.amp.autocast():
            logits = network(obs, valid)

            zeros = torch.zeros_like(logits)
            policy_loss = -(logits0.float().exp()*logits).where(valid, zeros).sum(axis=-1)[mask].mean()

            loss = policy_loss 

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        print('Stepped')

if __name__ == '__main__':
    run()