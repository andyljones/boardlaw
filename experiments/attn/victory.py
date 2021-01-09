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

def run(D=32, B=8*1024, T=5000, device='cuda'):

    worlds = Hex.initial(B, boardsize=9)

    boardsize = worldfunc(1).board.shape[-1]
    model = common.AttnModel(common.PosActions, boardsize, D).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    with tqdm(total=T) as pbar:
        for t, b in zip(range(T), batchgen(worlds, B)):
            outputs = model(b.worlds.obs)

            loss = F.nll_loss(outputs.reshape(B, -1), b.actions)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.update(1)
            pbar.set_description(f'{loss:.2f}')
