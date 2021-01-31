import aljpy.plot
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
        path = f'/tmp/batches/{run}.pkl'
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, 'wb+') as f:
            for data in r.iter_content(chunk_size=1024*1024):
                f.write(data)
                pbar.update(len(data)/(1024*1024))

def trained_path(run):
    from portalocker import RLock
    with RLock('/tmp/_lock'):
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

def run(name, width, depth, batch, lr, T=np.inf):
    set_devices()
    full = load_trained('2021-01-24 20-30-48 muddy-make')
    train, test = full[:1023], full[-1]
    obs_test, seats_test, y_test = decompress(test)

    network = models.ConvModel(obs_test.size(1), width=width, depth=depth).cuda()
    opt = torch.optim.Adam(network.parameters(), lr=lr)

    stats = []
    mult = 32*1024//batch
    for t, (obs, seats, y) in enumerate(split(cycle(train), mult)):
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

        if (t > 500*mult) & (t % 100 == 0):
            diff = (pd.DataFrame(stats)['train']
                        .ewm(span=100).mean()
                        .pipe(lambda df: df.iloc[-500*mult] - df.iloc[-1]))
            if abs(diff) < .005:
                break

        if t == T:
            break

    df = pd.DataFrame(stats)
    path = ROOT / 'results' / name / f'{width}n{depth}l{batch}b{lr/1e-4:.0f}lr.csv'
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path)

def load_basic_results(name):
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

def plot_envelope(aug, xlabel, ax=None, base=2, legend=False):
    x = aug.pivot('depth', 'width', xlabel)
    y = 1 - aug.pivot('depth', 'width', 'rv')
    colors = plt.cm.viridis(np.linspace(0, 1, x.shape[1]))
    _, ax = plt.subplots(1, 1) if ax is None else (None, ax)
    ax.set_xscale('log', basex=base)
    for c, color in zip(x.columns, colors):
        ax.plot(x[c], y[c], marker='.', label=c, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('var explained')
    ax.grid(True)
    aljpy.plot.percent_axis(ax, axis='y')
    if legend:
        ax.legend(title='width')
    return ax

def plot_results(raw=None):
    raw = load_opt_results('fc-opt') if raw is None else raw

    time = raw.xs('time', 1, 4).pipe(lambda df: df - df.iloc[0]).max()
    train = raw.xs('train', 1, 4).ewm(span=100).mean().min()
    steps = raw.xs('train', 1, 4).pipe(lambda df: df.ffill().where(df.bfill().notnull())).notnull().sum()
    picks = train.groupby(['n', 'l']).idxmin()

    aug = (train
            .groupby(['n', 'l']).min()
            .reset_index())
    aug.columns = ['width', 'depth', 'rv']

    aug['time'] = time[picks].values
    aug['steps'] = steps[picks].values

    aug = (aug
            .assign(params=lambda df: (9*df.width**2 + df.width)*df.depth)
            .assign(memory=lambda df: 11*11*df.width*df.depth)
            .assign(flops=lambda df: 11*11*(9*df.width**3 + df.width)*df.depth))

    aug['total_flops'] = aug.flops*aug.steps

    fig, axes = plt.subplots(2, 2, sharey=True)
    plot_envelope(aug, 'depth', axes[0, 0], base=2)
    plot_envelope(aug, 'params', axes[0, 1])
    plot_envelope(aug, 'memory', axes[1, 0], legend=True)
    plot_envelope(aug, 'total_flops', axes[1, 1])
    axes[0, 1].set_ylabel('')
    axes[1, 1].set_ylabel('')
    fig.set_size_inches(12, 12)
    
    title = (
        'performance envelopes for predicting outcome of 11x11 Hex games with conv networks\n'
        '(width doubles line-to-line, depth doubles node-to-node)')
    fig.suptitle(title, y=.95)

def plot_bests(raw):
    raw = load_opt_results('fc-opt') if raw is None else raw

    time = raw.xs('time', 1, 4).pipe(lambda df: df - df.iloc[0])
    train = raw.xs('train', 1, 4).pipe(lambda df: df.ewm(span=100).mean().where(df.notnull()))

    best_time = {}
    for t in np.arange(60, 2000, 60):
        best_time[t] = train.where(time < t).min().idxmin()
    best_time = pd.concat({r: pd.Series(train[r].values, pd.to_timedelta(time[r].values, unit='s')).dropna().resample('15s').mean() for r in best_time.values() if r in train}, 1)
    best_time.columns.names = ('w', 'l', 'b', 'lr')

    best_steps = {}
    for s in np.arange(100, 7000, 100):
        best_steps[s] = train[:s].min().idxmin()
    best_steps = pd.concat({r: train[r] for r in best_steps.values() if r in train}, 1)

    fig, axes = plt.subplots(2, 1)

    best_time.plot(ax=axes[0], title='best by time')
    best_steps.plot(ax=axes[1], title='best by num of steps')

def demo():
    import jittens
    widths = [1, 2, 4, 8, 16, 32, 64, 128]
    depths = [1, 2, 4, 8, 16, 32, 64, 128]
    batches = [16*1024]
    lrs = [8e-4]
    for width in widths:
        for depth in depths:
            for batch in batches:
                for lr in lrs:
                    jittens.jobs.submit(f'python -c "from experiments.architecture import *; run(\'conv-opt\', {width}, {depth}, {batch}, {lr})" >logs.txt 2>&1', dir='.', resources={'gpu': 1})

    while not jittens.finished():
        jittens.manage()