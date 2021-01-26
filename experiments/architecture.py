import pandas as pd
import pickle
from tqdm.auto import tqdm
import numpy as np
from boardlaw.main import worldfunc, agentfunc, mix, half
from boardlaw.learning import reward_to_go
from pavlov import storage
from rebar import arrdict
import torch
from logging import getLogger
from boardlaw import heads
from torch import nn
from torch.nn import functional as F
from IPython.display import clear_output
from itertools import cycle

log = getLogger(__name__)

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
        return heads.scatter_values(torch.tanh(v), seats)

def compress(obs, seats, y):
    return {
        'obs': np.packbits(obs.bool().cpu().numpy()),
        'obs_shape': obs.shape,
        'seats': seats.bool().cpu().numpy(),
        'y': y.cpu().numpy()}

def decompress(comp):
    raw = np.unpackbits(comp['obs'])
    obs = torch.as_tensor(raw.reshape(comp['obs_shape'])).cuda().float()
    seats = torch.as_tensor(comp['seats']).cuda().int()
    y = torch.as_tensor(comp['y']).cuda().float()
    return obs, seats, y

def experience(run, n_envs=32*1024, device='cuda'):
    #TODO: Restore league and sched when you go back to large boards
    worlds = mix(worldfunc(n_envs, device=device))
    agent = agentfunc(device)

    sd = storage.load_latest(run)
    agent.load_state_dict(sd['agent'])

    # Collect experience
    buffer = []
    while True:
        with torch.no_grad():
            decisions = agent(worlds, value=True)
        new_worlds, transition = worlds.step(decisions.actions)

        buffer.append(arrdict.arrdict(
            obs=worlds.obs.byte(),
            seats=worlds.seats,
            v=decisions.v,
            terminal=transition.terminal,
            rewards=transition.rewards).detach())

        if len(buffer) > worlds.boardsize**2:
            buffer = buffer[1:]
            chunk = arrdict.stack(buffer)
            terminal = torch.stack([chunk.terminal for _ in range(worlds.n_seats)], -1)
            targets = reward_to_go(
                        chunk.rewards.float(), 
                        chunk.v.float(), 
                        terminal)
            
            yield chunk.obs[0], chunk.seats[0], targets[0]
        else:
            if len(buffer) % worlds.boardsize == 0:
                log.info(f'Experience: {len(buffer)}/{worlds.boardsize**2}')

        worlds = new_worlds

def save(count=1024):
    buffer = []
    for obs, seats, y in tqdm(experience('*muddy-make'), total=count):    
        buffer.append(compress(obs, seats, y))

        if len(buffer) == count:
            with open('output/architecture-batches.pkl', 'wb+') as f:
                pickle.dump(buffer, f)
            break

def load():
    with open('output/architecture-batches.pkl', 'rb+') as f:
        compressed = pickle.load(f)

    np.random.seed(0)
    return np.random.permutation(compressed)

def residual_var(y, yhat):
    return (y - yhat).pow(2).mean().div(y.pow(2).mean()).item()

def report(stats):
    last = pd.DataFrame(stats).ffill().iloc[-1]
    clear_output(wait=True)
    print(
        f'step  {len(stats)}\n'
        f'train {last.train:.2f}\n'
        f'test  {last.test:.2f}\n'
        f'gap   {last.train - last.test:.2f}')

def plot(stats):
    pd.DataFrame(stats).applymap(float).ewm(span=20).mean().ffill().plot()

def run():
    network = FCModel(worldfunc(1).boardsize).cuda()

    opt = torch.optim.Adam(network.parameters(), lr=1e-2)

    full = load()
    train, test = full[:1023], full[-1]
    obs_test, seats_test, y_test = decompress(test)
    stats = []
    for i, comp in enumerate(cycle(train)):
        obs, seats, y = decompress(comp)
        yhat = network(obs, seats)

        loss = (y - yhat).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        stat = {'train': residual_var(y, yhat), 'test': np.nan}
        if i % 100 == 0:
            res_var_test = residual_var(y_test, network(obs_test, seats_test).detach())
            stat['test'] = res_var_test
        stats.append(stat)
            
        if i % 10 == 0:
            report(stats)

    plot(stats)