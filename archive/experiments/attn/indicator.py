import torch
from torch.nn import functional as F
from torch import nn
from . import common
from tqdm.auto import tqdm

def pointer_loss(rows, cols, boardsize, outputs):
    targets = 2*torch.stack([rows/boardsize, cols/boardsize], -1) - 1
    return (targets - outputs).pow(2).mean()

def action_loss(rows, cols, boardsize, outputs):
    B = rows.shape[0]
    targets = rows*boardsize + cols
    return F.nll_loss(outputs.reshape(B, -1), targets)

class Mechanical(nn.Module):

    def __init__(self, boardsize, D):
        super().__init__()

    def forward(self, obs):
        x = 10*(obs.sum(-1) - 1) - 1
        x = x.reshape(obs.shape[0], -1)
        x = F.log_softmax(x, -1)
        x = x.reshape(obs.shape[:-1])
        return x

def run(T=5000, D=32, B=8*1024, boardsize=9, device='cuda'):
    model = common.AttnModel(common.PosActions, boardsize, D).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    envs = torch.arange(B, device=device)

    with tqdm(total=T) as pbar:
        for t in range(T):
            obs = torch.zeros((B, boardsize, boardsize, 2), device=device)

            r = torch.randint(0, boardsize, size=(B,), device=device)
            c = torch.randint(0, boardsize, size=(B,), device=device)
            obs[envs, r, c, 1] = 1.

            x = model(obs)
            loss = action_loss(r, c, boardsize, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_description(f'{loss:.2f}')
            if t % 10 == 0:
                pbar.update(10)
            if loss < 1e-2:
                print(f'Finished in {t} steps')
                break
