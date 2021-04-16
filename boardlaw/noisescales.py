from rebar import parallel
import pandas as pd
import torch
from .arena import common
from boardlaw import learning, main, sql
from rebar import arrdict, dotdict
import numpy as np
from tqdm.auto import tqdm
from logging import getLogger
from multiprocessing import set_start_method

log = getLogger(__name__)

# Smallest networks to make it to within -.5 of perfect play.
RUNS = {
    3: '2021-02-17 19-34-03 valid-ships',
    4: '2021-02-21 05-22-28 watery-drunks',
    5: '2021-02-19 11-15-16 skinny-tactic',
    6: '2021-02-21 08-52-28 these-plow',
    7: '2021-02-19 19-50-36 tan-buffer',
    8: '2021-02-21 22-55-52 double-spoon',
    9: '2021-02-20 21-11-32 intent-nets'}

def stored_agent(agent_id):
    info = sql.query('select * from agents_details where id == ?', int(agent_id)).iloc[0]
    agent = common.agent(info.run, info.idx, 'cuda')
    agent.kwargs['n_nodes'] = info.test_nodes
    agent.kwargs['c_puct'] = info.test_c
    return agent

def stored_worlds(agent_id, n_envs):
    info = sql.query('select * from agents_details where id == ?', int(agent_id)).iloc[0]
    return common.worlds(info.run, n_envs, 'cuda')

def collect(agent_id, n_envs=32*1024):
    agent = stored_agent(agent_id)
    worlds = stored_worlds(agent_id, n_envs)

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

def flat_grad(network, loss):
    for p in network.parameters():
        p.grad = None
    loss.backward(retain_graph=True)
    return torch.cat([p.grad.flatten() for p in network.parameters() if p.grad is not None]) 

def gradient(network, batch):
    d0 = batch.decisions
    d = network(batch.worlds)

    zeros = torch.zeros_like(d.logits)
    l = d.logits.where(d.logits > -np.inf, zeros)
    l0 = d0.logits.float().where(d0.logits > -np.inf, zeros)

    policy_loss = -(l0.exp()*l).sum(axis=-1).mean()

    target_value = batch.reward_to_go
    value_loss = (target_value - d.v).square().mean()

    return arrdict.arrdict(
        policy=flat_grad(network, policy_loss),
        value=flat_grad(network, value_loss),
        joint=flat_grad(network, policy_loss + value_loss))

def gradients(network, chunk):
    grads = []
    for t in range(chunk.reward_to_go.size(0)):
        grads.append(gradient(network, chunk[t]))
    return arrdict.stack(grads)

def noise_scale_components(chunk, gs, kind, agent_id):
    T = chunk.reward_to_go.size(0)
    B = chunk.reward_to_go.size(1)
    return pd.Series({
        'agent_id': agent_id,
        'kind': kind,
        'mean_sq': float(gs.mean(0).pow(2).mean()),
        'sq_mean': float(gs.pow(2).mean()),
        'variance': float((gs - gs.mean(0, keepdim=True)).pow(2).mean(0).mul(T/(T-1)).mean(0)),
        'n_params': float(gs.shape[1]),
        'batch_size': float(B),
        'batches': float(T)})

def noise_scale(result):
    return result.batch_size*result.variance/result.mean_sq

def evaluate_noise_scale(agent_id):
    extant = sql.query('select * from noise_scales where agent_id == ?', int(agent_id))
    if len(extant) == 0:
        agent, chunk = collect(agent_id)
        gs = gradients(agent.network, chunk)

        results = pd.DataFrame([noise_scale_components(chunk, gs[k], k, agent_id) for k in gs])
        log.info(f'{agent_id}: {noise_scale(results.iloc[0]):.0f}')
        sql.save_noise_scale(results)

def agents_opponent(agent_id):
    return sql.query('''
        select id from agents 
        where snap == (
            select snap from agents where id == ?)
        and nodes == 64 
        and c == 1./16''', int(agent_id)).id.iloc[0]
    
def evaluate_perf(agent_id, n_envs=1024):
    opponent_id = agents_opponent(agent_id)
    extant = sql.query('''
        select * from trials 
        where ((black_agent == ?) and (white_agent == ?)) 
        or ((white_agent == ?) and (black_agent == ?))''', 
        int(agent_id), int(opponent_id), int(agent_id), int(opponent_id))
    games = (extant.black_wins + extant.white_wins).sum()
    if games < n_envs:
        a = stored_agent(agent_id)
        o = stored_agent(opponent_id)
        w = stored_worlds(agent_id, n_envs)

        results = common.evaluate(w, [(agent_id, a), (opponent_id, o)])

        sql.save_trials(results)

def evaluate(run, idx, nodes, c_puct):
    snap_id = sql.query_one('select id from snaps where run == ? and idx == ?', run, int(idx)).id
    extant = sql.query('select * from agents where snap == ? and nodes == ? and c == ?', int(snap_id), int(nodes), float(c_puct))
    if len(extant) == 0:
        log.info(f'Creating agent run="{run}", idx={idx}, nodes={nodes}, c_puct={c_puct:.3f}')
        sql.execute('insert into agents values (null, ?, ?, ?)', int(snap_id), int(nodes), float(c_puct))
        extant = sql.query('select * from agents where snap == ? and nodes == ? and c == ?', int(snap_id), int(nodes), float(c_puct))
    agent_id = extant.id.iloc[0]

    evaluate_noise_scale(agent_id)
    evaluate_perf(agent_id)

def sweep_trees(boardsize=None):
    if boardsize is None:
        for b in RUNS:
            sweep_trees(b)
        return 
    run = RUNS[boardsize]
    snaps = sql.query('select * from snaps where run == ?', run)

    set_start_method('spawn', True)
    with parallel.parallel(evaluate, N=2, executor='cuda', desc=str(boardsize)) as pool:
        futures = {}
        for idx in snaps.idx[snaps.idx % 2 == 0].unique():
            for nodes in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for c in [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.]:
                    futures[idx, nodes, c] = pool(run, idx, nodes, c)
        pool.wait(futures)

def sweep_runs():
    runs = sql.query('select * from runs where description like "bee/%"')
    np.random.seed(0)
    runs.sample(frac=1)

    set_start_method('spawn', True)
    for i, run in enumerate(runs.run.unique()):
        snaps = sql.query('select * from snaps where run == ?', run)
        with parallel.parallel(evaluate, N=2, executor='cuda', desc=str(i)) as pool:
            pool.wait([pool(run, idx, 64, 1/16) for idx in snaps.idx.unique()])

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

def node_sweep(run='2021-02-21 09-04-51 wavy-mills', idx=13):
    results = {}
    for c in [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1., 2.]:
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            results[c, n] = noise_scale_components(run, idx, n_nodes=n, c_puct=c)
    results = pd.concat(results, 0).unstack()
    return results