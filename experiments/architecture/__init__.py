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
import portalocker

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

def upload():
    from boardlaw import backup
    backup.sync('output/experiments/architecture/batches', 'boardlaw-public:experiments/architecture/batches')

def download(run):
    import requests
    from tqdm.auto import tqdm
    url = f'https://f002.backblazeb2.com/file/boardlaw-public/experiments/architecture/batches/{run.replace(" ", "+")}.pkl'
    r = requests.get(url, stream=True)
    with tqdm(total=int(r.headers.get('content-length'))/(1024*1024)) as pbar:
        with open(f'/tmp/{run}', 'wb+') as f:
            for data in r.iter_content(chunk_size=1024*1024):
                f.write(data)
                pbar.update(len(data)/(1024*1024))

def trained_path(run):
    with portalocker.FileLock('/tmp/_lock'):
        path = ROOT / 'batches' / f'{run}.pkl' 
        if path.exists():
            return path
        
        path = Path('/tmp') / 'batches' / f'{run}.pkl'
        if not path.exists():
            download(run)

        return path

def load_trained(run):
    path = trained_path(run)
        
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
        f'test  {last.test:.2f}', flush=True)

def plot(stats):
    pd.DataFrame(stats).applymap(float).ewm(span=20).mean().ffill().plot()

def run(name, width, depth, batch, lr, T=np.inf):
    set_devices()
    full = load_trained('2021-01-24 20-30-48 muddy-make')
    train, test = full[:1023], full[-1]
    obs_test, seats_test, y_test = decompress(test)

    network = models.FCModel(obs_test.size(1), width=width, depth=depth).cuda()
    opt = torch.optim.Adam(network.parameters(), lr=lr)

    stats = []
    for t, (obs, seats, y) in enumerate(split(cycle(train), 32*1024//batch)):
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
            if abs(diff) < .005:
                break

        if t == T:
            break

    df = pd.DataFrame(stats)
    path = ROOT / 'results' / name / f'{width}n{depth}l{batch}b{lr/1e-4:.0f}lr.csv'
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

def load_opt_results(name):
    results = {}
    for path in (ROOT / 'results' / name).glob('*.csv'):
        n, l, b, lr = re.match(r'(\d+)n(\d+)l(\d+)b(\d+)lr.csv', path.name).group(1, 2, 3, 4)
        results[int(n), int(l), int(b), float(lr)*1e-4] = pd.read_csv(path, index_col=0)
    df = pd.concat(results, 1)
    df.columns.names = ('n', 'l', 'b', 'lr', 'field')
    return df

def plot_envelope(aug, xlabel, ax=None):
    x = aug.pivot('width', 'depth', xlabel)
    y = aug.pivot('width', 'depth', 'rv')
    colors = plt.cm.viridis(np.linspace(0, 1, x.shape[1]))
    _, ax = plt.subplots(1, 1) if ax is None else (None, ax)
    ax.set_xscale('log', basex=10)
    for c, color in zip(x.columns, colors):
        ax.plot(x[c], y[c], marker='.', label=c, color=color)
    ax.legend(title='depth')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('resid var')
    ax.grid(True)
    ax.set_ylim(None, 1)


def plot_results():
    df = load_results('fc').xs('train', 1, 2).ewm(span=100).mean().min().unstack(1)

    steps = load_results('fc').xs('time', 1, 2).notnull().sum().unstack(1)
    time = load_results('fc').xs('time', 1, 2).pipe(lambda df: df - df.iloc[0]).max().unstack(1)
    df = pd.concat({'rv': df, 'time': time, 'steps': steps}, 1)
    df.columns.names = ('field', 'depth')
    df.index.name = 'width'

    aug = (df
        .stack()
        .reset_index()
        .assign(params=lambda df: (df.width**2 + df.width)*df.depth)
        .assign(memory=lambda df: df.width*df.depth)
        .assign(flops=lambda df: (df.width**3 + df.width)*df.depth))


    with plt.style.context('seaborn-poster'):
        mpl.rcParams['legend.title_fontsize'] = 'xx-large'
        fig, axes = plt.subplots(3, 2, sharey=True)
        plot_envelope(aug, 'width', axes[0, 0])
        plot_envelope(aug, 'params', axes[0, 1])
        plot_envelope(aug, 'flops', axes[1, 1])
        plot_envelope(aug, 'memory', axes[1, 0])
        plot_envelope(aug, 'time', axes[2, 0])
        plot_envelope(aug, 'steps', axes[2, 1])
        axes[0, 1].set_ylabel('')
        axes[1, 1].set_ylabel('')
        axes[2, 1].set_ylabel('')
        fig.set_size_inches(16, 24)
        fig.suptitle('performance envelopes for predicting outcome of 11x11 Hex games with fully-connected networks\n(depth doubles line-to-line, width doubles node-to-node)')

def demo():
    import jittens
    widths = [1, 8, 64, 512]
    depths = [1, 8, 64, 512]
    batches = [1024, 4*1024, 16*1024]
    lrs = [1e-4, 8e-4, 64e-4, 512e-4]
    for width in widths:
        for depth in depths:
            for batch in batches:
                for lr in lrs:
                    jittens.submit(f'python -c "from experiments.architecture import *; run(\'fc-opt\', {width}, {depth}, {batch}, {lr})" >logs.txt 2>&1', dir='.', resources={'gpu': 1})

    while not jittens.finished():
        jittens.manage()