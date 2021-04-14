import pandas as pd
import torch
from .arena import common
from boardlaw import learning, main, sql
from rebar import arrdict, dotdict
import numpy as np
from tqdm.auto import tqdm

def collect(run, idx, n_envs=32*1024):
    agent = common.agent(run, idx, 'cuda')
    worlds = common.worlds(run, n_envs, 'cuda')

    buffer = []
    while True:
        while len(buffer) < 64:
            with torch.no_grad():
                decisions = agent(worlds, value=True)
            new_worlds, transition = worlds.step(decisions.actions)

            buffer.append(arrdict.arrdict(
                worlds=worlds,
                decisions=decisions.half(),
                transitions=learning.half(transition)).detach())

            worlds = new_worlds

        chunk, buffer = main.as_chunk(buffer, n_envs)
        
        mixness = chunk.transitions.terminal.float().mean(1)
        mixness = (mixness.max() - mixness.min())/mixness.median()
        if mixness < .25:
            break

    return agent, chunk

def gradient(network, batch):
    d0 = batch.decisions
    d = network(batch.worlds)

    zeros = torch.zeros_like(d.logits)
    l = d.logits.where(d.logits > -np.inf, zeros)
    l0 = d0.logits.float().where(d0.logits > -np.inf, zeros)

    policy_loss = -(l0.exp()*l).sum(axis=-1).mean()

    target_value = batch.reward_to_go
    value_loss = (target_value - d.v).square().mean()

    loss = policy_loss + value_loss

    for p in network.parameters():
        p.grad = None
    loss.backward()

    return torch.cat([p.grad.flatten() for p in network.parameters() if p.grad is not None]) 

def gradients(network, chunk):
    grads = []
    for t in range(chunk.reward_to_go.size(0)):
        grads.append(gradient(network, chunk[t]))
    return torch.stack(grads)

def noise_scale(run, idx):
    agent, chunk = collect(run, idx)

    gs = gradients(agent.network, chunk)
    gb = gs.mean(0)

    G2s = gs.pow(2).mean()
    G2b = gb.pow(2).mean()

    T = chunk.reward_to_go.size(0)
    B = chunk.reward_to_go.size(1)
    Bb = T*B
    Bs = B

    G2 = 1/(Bb - Bs) * (Bb*G2b - Bs*G2s)
    S = 1/(1/Bs - 1/Bb) * (G2s - G2b)

    return pd.Series({
        'G2s': float(G2s),
        'G2b': float(G2b),
        'Bs': float(B),
        'Bb': float(T*B),
        'S': float(S), 
        'G2': float(G2)})

def run(n=2, r=0):
    extant = sql.query('select id from noise_scales').id
    desired = (sql.agent_query()
                .groupby('snap_id').first()
                .assign(params=lambda df: df.width**2 * df.depth)
                .sample(frac=1)
                .drop(extant, axis=0))

    desired = desired[desired.index % n == r]

    for id, row in tqdm(list(desired.iterrows())):
        result = noise_scale(row.run, row.idx)
        result['id'] = id
        sql.save_noise_scale(result)