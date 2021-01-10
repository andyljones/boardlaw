import numpy as np
import torch
from torch import nn
from rebar import arrdict
import pickle
from pathlib import Path
from . import common
from tqdm.auto import tqdm
from boardlaw.hex import Hex, color_obs
from torch.nn import functional as F
from boardlaw.main import worldfunc, agentfunc
from boardlaw.hex import Hex
from boardlaw.hex import plot_board
import matplotlib.pyplot as plt
from rebar.recording import Encoder


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

def plot_soln(i, obs, targets, outputs, attns=None):
    boardsize = obs.shape[-2]

    fig, axes = plt.subplots(2, 2)

    ax = axes[0, 0]
    probs = outputs[i].detach().cpu().exp().numpy()
    plot_board(plt.cm.viridis(probs), ax=ax)
    ax.set_title('probs')

    ax = axes[0, 1]
    ts = targets[i].detach().cpu().numpy().reshape(boardsize, boardsize)
    plot_board(plt.cm.viridis(ts), ax=ax)
    ax.set_title('targets')

    ax = axes[1, 0]
    plot_board(color_obs(obs[i].cpu().numpy()), ax=ax)
    ax.set_title('board')

    ax = axes[1, 1]
    if attns is None:
        ax.axis('off')
    else:
        attn = attns.detach().cpu().numpy()[i].max(0).max(-1)
        attn = attn.reshape(boardsize, boardsize)
        plot_board(plt.cm.viridis(attn/attn.max()), ax=ax)
        ax.set_title('attn')

def animate(i, obs, attns):
    obs = obs[i].detach().cpu().numpy()
    attn = attns[i].detach().cpu().numpy()
    boardsize = int(attn.shape[1]**.5)
    attn = attn.transpose(0, 2, 1).reshape(attn.shape[0], attn.shape[2], boardsize, boardsize)

    rows, cols = attn.shape[:2]

    with Encoder(fps=1) as enc:
        for r in range(rows+3):
            r = min(r, rows-1)
            fig, axes = plt.subplots(1, cols+1, squeeze=False)
            fig.set_size_inches(8*(cols+1), 8)

            plot_board(color_obs(obs), ax=axes[0, 0])
            for c in range(cols):
                colors = plt.cm.viridis(attn[r, c])
                plot_board(colors, ax=axes[0, c+1])
            enc(fig)
            plt.close(fig)
    enc.notebook()

def run(D=32, B=8*1024, T=5000, device='cuda'):

    worlds = Hex.initial(B, boardsize=7)

    boardsize = worlds.boardsize
    model = common.FCModel(common.PosActions, boardsize, D).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    with tqdm(total=T) as pbar:
        for t, b in zip(range(T), batchgen(worlds, B)):
            targets = terminal_actions(b.worlds)
            outputs = model(b.worlds.obs)

            infs = torch.full_like(targets, -np.inf)
            loss = -outputs.reshape(B, -1).where(targets == 1., infs).max(-1).values.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.update(1)
            pbar.set_description(f'{loss:.2f}')
