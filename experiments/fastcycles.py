from pathlib import Path
import pickle
from boardlaw.arena import evaluator
from pavlov import storage
from boardlaw.main import worldfunc, agentfunc, mix, half, as_chunk, optimize
import torch
from rebar import arrdict
from logging import getLogger
from itertools import combinations, combinations_with_replacement, product
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns

log = getLogger(__name__)

def clone(sd):
    return {k: v.clone().detach() for k, v in sd.items()}

def generate_state_dicts(run):
    n_envs = 24*1024
    buffer_len = 64
    device = 'cuda'

    #TODO: Restore league and sched when you go back to large boards
    worlds = mix(worldfunc(n_envs, device=device))
    agent = agentfunc(device)
    network = agent.network

    opt = torch.optim.Adam(network.parameters(), lr=1e-2, amsgrad=True)
    scaler = torch.cuda.amp.GradScaler()

    sd = storage.load_latest(run)
    agent.load_state_dict(sd['agent'])
    opt.load_state_dict(sd['opt'])

    state_dicts = [clone(network.state_dict())]

    buffer = []
    #TODO: Upgrade this to handle batches that are some multiple of the env count
    idxs = (torch.randint(buffer_len, (n_envs,), device=device), torch.arange(n_envs, device=device))
    for _ in range(8):

        # Collect experience
        while len(buffer) < buffer_len:
            with torch.no_grad():
                decisions = agent(worlds, value=True)
            new_worlds, transition = worlds.step(decisions.actions)

            buffer.append(arrdict.arrdict(
                worlds=worlds,
                decisions=decisions.half(),
                transitions=half(transition)).detach())

            worlds = new_worlds

            log.info(f'({len(buffer)}/{buffer_len}) actor stepped')

        # Optimize
        chunk, buffer = as_chunk(buffer, n_envs)
        optimize(network, scaler, opt, chunk[idxs])
        log.info('learner stepped')

        state_dicts.append(clone(network.state_dict()))

    return state_dicts 

def evaluate(pair, n_envs=256, device='cuda'):
    agents = {}
    for name, sd in pair.items():
        agent = agentfunc(device)
        agent.network.load_state_dict(sd)
        agents[name] = agent

    worlds = mix(worldfunc(n_envs, device=device))
    return evaluator.evaluate(worlds, agents)

def run():
    sds = generate_state_dicts('*muddy-make')

    results = []
    for i, j in tqdm(list(combinations_with_replacement(range(len(sds)), 2))):
        results.extend(evaluate({f'{i}.0': sds[i], f'{j}.1': sds[j]}))

    p = Path('output/experiments/fastcycles.pkl')
    p.parent.mkdir(exist_ok=True, parents=True)
    p.write_bytes(pickle.dumps(results))

    from collections import defaultdict
    summary = defaultdict(lambda: 0.)
    for r in results:
        blk = int(r.names[0][0])
        wht = int(r.names[1][0])
        summary[blk, wht, 'blk'] += r['wins'][0]
        summary[blk, wht, 'wht'] += r['wins'][1]
    df = pd.Series(summary).unstack(2).pipe(lambda df: df.div(df.sum(1), 0)).unstack(1)

    sns.heatmap(df.wht, vmin=.4, vmax=.6, cmap='RdBu')

    return results