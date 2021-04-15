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

def noise_scale_components(run, idx):
    agent, chunk = collect(run, idx)

    gs = gradients(agent.network, chunk)

    T = chunk.reward_to_go.size(0)
    B = chunk.reward_to_go.size(1)

    return pd.Series({
        'mean_sq': float(gs.mean(0).pow(2).mean()),
        'sq_mean': float(gs.pow(2).mean()),
        'variance': float((gs - gs.mean(0, keepdim=True)).pow(2).mean(0).mul(T/(T-1)).mean(0)),
        'n_params': float(gs.shape[1]),
        'batch_size': float(B),
        'batches': float(T)})

def noise_scale(meansq, var, **kwargs):
    return meansq/var

def run(n=2, r=0):
    extant = sql.query('select id from noise_scales').id
    desired = (sql.agent_query()
                .groupby('snap_id').first()
                .assign(params=lambda df: df.width**2 * df.depth)
                .sample(frac=1)
                .drop(extant, axis=0))

    desired = desired[desired.index % n == r]

    for id, row in tqdm(list(desired.iterrows())):
        result = noise_scale_components(row.run, row.idx)
        result['id'] = id
        sql.save_noise_scale(result)

def load():
    from analysis import data

    ags = data.load()

    noise = sql.query('''select * from noise_scales''')
    df = pd.merge(ags, noise, left_on='snap_id', right_on='id').query('test_nodes == 64')

    # High-var low-bias estimate
    Bb = df.batches*df.batch_size 
    Bs = df.batch_size
    Gb = df.mean_sq
    Gs = df.sq_mean

    G2 = 1/(Bb - Bs)*(Bb*Gb - Bs*Gs)
    S = 1/(1/Bs - 1/Bb)*(Gs - Gb)

    df['low_bias'] = S/G2
    
    # Low-var high-bias estimate
    df['low_var'] = df.batch_size*df.variance/df.mean_sq

    return df