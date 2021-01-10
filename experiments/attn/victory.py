import numpy as np
import torch
from torch import nn
from rebar import arrdict
import pickle
from pathlib import Path
from . import common
from tqdm.auto import tqdm
from boardlaw.hex import Hex
from torch.nn import functional as F
from boardlaw.main import worldfunc, agentfunc
from boardlaw.hex import Hex

def gamegen(worlds):
    while True:
        # with torch.no_grad():
        #     decisions = agent(worlds)
        actions = torch.distributions.Categorical(probs=worlds.valid.float()).sample()
        new_worlds, transitions = worlds.step(actions)

        if transitions.terminal.any():
            yield arrdict.arrdict(
                worlds=worlds[transitions.terminal],
                actions=actions[transitions.terminal],
                rewards=transitions.rewards[transitions.terminal])

        worlds = new_worlds

def batchgen(worlds, B):
    buffer = []
    for x in gamegen(worlds):
        buffer.append(x)
        n = sum(b.actions.size(0) for b in buffer)
        if n >= B:
            buffer = arrdict.cat(buffer)
            yield buffer[:B]
            buffer = [buffer[B:]] if buffer.actions.size(0) > B else []

def terminal_actions(worlds):
    terminal = torch.zeros((worlds.n_envs, worlds.boardsize**2), device=worlds.device).bool()
    for a in torch.arange(worlds.boardsize**2, device=worlds.device):
        actions = torch.full((worlds.n_envs,), a, device=worlds.device)
        mask = worlds.valid.gather(1, actions[:, None]).squeeze(-1)
        _, transitions = worlds[mask].step(actions[mask])
        terminal[mask, a] = transitions.terminal
    return terminal.float()

def plot(worlds, targets, outputs, i):
    import matplotlib.pyplot as plt
    from boardlaw.hex import plot_board

    fig, (l, m, r) = plt.subplots(1, 3)

    probs = outputs[i].detach().cpu().exp()
    colors = np.stack(np.vectorize(plt.cm.viridis)(probs), -1)
    plot_board(colors, ax=l)

    ts = targets[i].detach().cpu().numpy().reshape(worlds.boardsize, worlds.boardsize)
    colors = np.stack(np.vectorize(plt.cm.viridis)(ts), -1)
    plot_board(colors, ax=m)

    worlds.display(i, ax=r, colors='board')

    return fig

def run(D=32, B=8*1024, T=5000, device='cuda'):

    worlds = Hex.initial(B, boardsize=7)

    boardsize = worlds.boardsize
    model = common.FCModel(common.PosActions, boardsize, D).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    with tqdm(total=T) as pbar:
        for t, b in zip(range(T), batchgen(worlds, B)):
            targets = terminal_actions(b.worlds)
            outputs = model(b.worlds.obs)

            offset = (targets.sum(-1)*targets.sum(-1).log()).mean()
            loss = -outputs.reshape(B, -1).mul(targets).sum(-1).mean() - offset

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.update(1)
            pbar.set_description(f'{loss:.2f}')
