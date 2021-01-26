import pandas as pd
import pickle
from tqdm.auto import tqdm
import numpy as np
from boardlaw.main import mix
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

log = getLogger(__name__)

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
    path = Path(f'output/experiments/architecture/batches/{run}.pkl')
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, 'wb+') as f:
        pickle.dump(buffer, f)

def load_trained(run):
    run = runs.resolve(run)
    with open(f'output/experiments/architecture/batches/{run}.pkl', 'rb+') as f:
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
    clear_output(wait=True)
    print(
        f'step  {len(stats)}\n'
        f'train {last.train:.2f}')

def plot(stats):
    pd.DataFrame(stats).applymap(float).ewm(span=20).mean().ffill().plot()

def run():
    full = load_trained('*muddy-make')
    train, test = full[:1023], full[-1]
    obs_test, seats_test, y_test = decompress(test)

    network = models.FCModel(obs_test.size(1)).cuda()
    opt = torch.optim.Adam(network.parameters(), lr=1e-2)

    stats = []
    for i, (obs, seats, y) in enumerate(split(cycle(train), 1)):
        yhat = network(obs, seats)

        loss = (y - yhat).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        stat = {'train': residual_var(y, yhat), 'test': np.nan}
        if i % 100 == 0:
            res_var_test = residual_var(y_test, network(obs_test, seats_test))
            stat['test'] = res_var_test
        stats.append(stat)
            
        if i % 10 == 0:
            report(stats)

    plot(stats)