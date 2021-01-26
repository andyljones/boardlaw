import time
import matplotlib.pyplot as plt
import re
import pandas as pd
import pickle
from tqdm.auto import tqdm
import numpy as np
from boardlaw.main import mix, set_devices
from boardlaw.mcts import MCTSAgent
from boardlaw.hex import Hex
from boardlaw.learning import reward_to_go
from boardlaw.validation import RandomAgent
from pavlov import storage, runs
from rebar import arrdict
import torch
from logging import getLogger
from . import models
from IPython.display import clear_output
from itertools import cycle
from pathlib import Path
import matplotlib as mpl

log = getLogger(__name__)

ROOT = Path('output/experiments/architecture')

def generate(agent, worlds):
    buffer = []
    while True:
        with torch.no_grad():
            decisions = agent(worlds, value=True)
        new_worlds, transition = worlds.step(decisions.actions)

        buffer.append(arrdict.arrdict(
            obs=worlds.obs,
            seats=worlds.seats,
            v=decisions.v,
            terminal=transition.terminal,
            rewards=transition.rewards).detach())

        # Waiting till the buffer matches the boardsize guarantees every traj is terminated
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

def generate_random(boardsize, n_envs=32*1024, device='cuda'):
    agent = RandomAgent()
    worlds = mix(Hex.initial(n_envs, boardsize=boardsize, device=device))

    yield from generate(agent, worlds)

def generate_trained(run, n_envs=32*1024, device='cuda'):
    #TODO: Restore league and sched when you go back to large boards
    boardsize = runs.info(run)['boardsize']
    worlds = mix(Hex.initial(n_envs, boardsize=boardsize, device=device))

    network = storage.load_raw(run, 'model').cuda() 
    agent = MCTSAgent(network)
    agent.load_state_dict(storage.load_latest(run, device)['agent'])

    sd = storage.load_latest(run)
    agent.load_state_dict(sd['agent'])

    yield from generate(agent, worlds)

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

def save_trained(run, count=1024):
    buffer = []
    for obs, seats, y in tqdm(generate_trained(run), total=count):    
        buffer.append(compress(obs, seats, y))

        if len(buffer) == count:
            break

    run = runs.resolve(run)
    path = ROOT / 'batches' / '{run}.pkl'
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, 'wb+') as f:
        pickle.dump(buffer, f)

def load_trained(run):
    for root in [Path('/tmp'), ROOT]:
        path = root / 'batches' / f'{run}.pkl'
        if path.exists():
            break
    else:
        raise IOError()
        
    with open(path, 'rb+') as f:
        compressed = pickle.load(f)

    np.random.seed(0)
    return np.random.permutation(compressed)

def split(comps, chunks):
    for comp in comps:
        obs, seats, y = decompress(comp)
        yield from zip(obs.chunk(chunks, 0), seats.chunk(chunks, 0), y.chunk(chunks, 0))

def residual_var(y, yhat):
    return (y - yhat).pow(2).mean().div(y.pow(2).mean()).detach().item()

def report(stats):
    last = pd.DataFrame(stats).ffill().iloc[-1]
    print(
        f'step  {len(stats)}\n'
        f'train {last.train:.2f}\n'
        f'test  {last.test:.2f}')

def plot(stats):
    pd.DataFrame(stats).applymap(float).ewm(span=20).mean().ffill().plot()

def run(name, width, depth, T=np.inf):
    set_devices()
    full = load_trained('2021-01-24 20-30-48 muddy-make')
    train, test = full[:1023], full[-1]
    obs_test, seats_test, y_test = decompress(test)

    network = models.FCModel(obs_test.size(1), width=width, depth=depth).cuda()
    opt = torch.optim.Adam(network.parameters(), lr=1e-2)

    stats = []
    for t, (obs, seats, y) in enumerate(split(cycle(train), 1)):
        yhat = network(obs, seats)

        loss = (y - yhat).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        n = torch.nn.utils.clip_grad_norm_(network.parameters(), 1).item()
        opt.step()

        stat = {'train': residual_var(y, yhat), 'test': np.nan, 'n': n, 'time': time.time()}
        if t % 100 == 0:
            res_var_test = residual_var(y_test, network(obs_test, seats_test))
            stat['test'] = res_var_test
        stats.append(stat)
            
        if t % 100 == 0:
            report(stats)

        if (t > 1000) & (t % 100 == 0):
            diff = (pd.DataFrame(stats)['train']
                        .ewm(span=100).mean()
                        .pipe(lambda df: df.iloc[-1000] - df.iloc[-1]))
            if diff < .005:
                break

        if t == T:
            break

    df = pd.DataFrame(stats)
    path = ROOT / 'results' / name / f'{width}n{depth}l.csv'
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path)

def load_results(name):
    results = {}
    for path in (ROOT / 'results' / name).glob('*.csv'):
        n, l = re.match(r'(\d+)n(\d+)l.csv', path.name).group(1, 2)
        results[int(n), int(l)] = pd.read_csv(path, index_col=0)
    df = pd.concat(results, 1)
    df.columns.names = ('n', 'l', 'field')
    return df

def plot_envelope(aug, xlabel, ax=None):
    x = aug.pivot('width', 'depth', xlabel)
    y = aug.pivot('width', 'depth', 'rv')
    colors = plt.cm.viridis(np.linspace(0, 1, x.shape[1]))
    _, ax = plt.subplots(1, 1) if ax is None else (None, ax)
    ax.set_xscale('log', basex=2)
    for c, color in zip(x.columns, colors):
        ax.plot(x[c], y[c], marker='.', label=c, color=color)
    ax.legend(title='depth')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('resid var')
    ax.grid(True)


def plot_results():
    df = load_results().xs('train', 1, 2).ewm(span=100).mean().iloc[-1].unstack(1)
    df.columns.name = 'depth'
    df.index.name = 'width'

    aug = (df
        .stack()
        .rename('rv')
        .reset_index()
        .assign(params=lambda df: (df.width**2 + df.width)*df.depth)
        .assign(flops=lambda df: (df.width**3 + df.width)*df.depth))

    with plt.style.context('seaborn-poster'):
        mpl.rcParams['legend.title_fontsize'] = 'xx-large'
        fig, axes = plt.subplots(1, 2, sharey=True)
        plot_envelope(aug, 'params', axes[0])
        plot_envelope(aug, 'flops', axes[1])
        axes[1].set_ylabel('')
        fig.set_size_inches(16, 8)
        fig.suptitle('performance envelopes for predicting outcome of 11x11 Hex games with fully-connected networks\n(lines are varying width, constant depth)')

def demo():
    import jittens
    widths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    depths = [1, 2, 4, 8, 16, 32, 64, 128]
    for width in widths:
        for depth in depths:
            jittens.submit(f'python -c "from experiments.architecture import *; run({width}, {depth})" >logs.txt 2>&1', dir='.', resources={'gpu': 1})

    while not jittens.finished():
        jittens.manage()