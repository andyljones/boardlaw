import gc
import time
import numpy as np
import torch
from rebar import arrdict, profiling, pickle
from pavlov import stats, logs, runs, storage, archive
from . import hex, mcts, networks, learning, validation, analysis, arena, leagues
from torch.nn import functional as F
from logging import getLogger

log = getLogger(__name__)

@torch.no_grad()
def chunk_stats(chunk, n_new):
    with stats.defer():
        tail = chunk[-n_new:]
        d, t = tail.decisions, tail.transitions
        n_trajs = t.terminal.sum()
        n_inputs = t.terminal.size(0)
        n_samples = t.terminal.nelement()
        n_sims = d.n_sims.int().sum()
        stats.rate('sample-rate.actor', n_samples)
        stats.mean('traj-length', n_samples, n_trajs)
        stats.cumsum('count.traj', n_trajs)
        stats.cumsum('count.inputs', n_inputs)
        stats.cumsum('count.chunks', 1)
        stats.cumsum('count.samples', n_samples)
        stats.cumsum('count.sims', n_sims)
        stats.rate('step-rate.chunks', 1)
        stats.rate('step-rate.inputs', n_inputs)
        stats.rate('sim-rate', n_sims)
        stats.mean('mcts-n-leaves', d.n_leaves.float().mean())

        wins = (t.rewards == 1).sum(0).sum(0)
        for i, w in enumerate(wins):
            stats.mean(f'wins.seat-{i}', w, n_trajs)

        d, t = chunk.decisions, chunk.transitions
        v = d.v[t.terminal]
        w = t.rewards[t.terminal]
        stats.mean('progress.corr.terminal', ((v - v.mean())*(w - w.mean())).mean()/(v.var()*w.var())**.5)

        v = d.v[:-1][t.terminal[1:]]
        w = t.rewards[1:][t.terminal[1:]]
        stats.mean('progress.corr.penultimate', ((v - v.mean())*(w - w.mean())).mean()/(v.var()*w.var())**.5)

def as_chunk(buffer, buffer_inc):
    chunk = arrdict.stack(buffer)
    terminal = torch.stack([chunk.transitions.terminal for _ in range(chunk.worlds.n_seats)], -1)
    chunk['reward_to_go'] = learning.reward_to_go(
        chunk.transitions.rewards.float(), 
        chunk.decisions.v.float(), 
        terminal).half()
    chunk_stats(chunk, buffer_inc)
            
    buffer = buffer[buffer_inc:]

    return chunk, buffer

def rel_entropy(logits):
    valid = (logits > -np.inf)
    zeros = torch.zeros_like(logits)
    logits = logits.where(valid, zeros)
    probs = logits.exp().where(valid, zeros)
    return (-(logits*probs).sum(-1).mean(), torch.log(valid.sum(-1).float()).mean())

def optimize(network, scaler, opt, batch):
    mask = batch.is_prime
    w, d0, t = batch.worlds[mask], batch.decisions[mask], batch.transitions[mask]

    with torch.cuda.amp.autocast():
        d = network(w)

        zeros = torch.zeros_like(d.logits)
        l = d.logits.where(d.logits > -np.inf, zeros)
        l0 = d0.logits.float().where(d0.logits > -np.inf, zeros)

        policy_loss = -(l0.exp()*l).sum(axis=-1).mean()

        target_value = batch.reward_to_go[mask]
        value_loss = (target_value - d.v).square().mean()

        loss = policy_loss + value_loss

    old = torch.cat([p.flatten() for p in network.parameters()])

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    new = torch.cat([p.flatten() for p in network.parameters()])

    with stats.defer():
        #TODO: Contract these all based on late-ness
        stats.mean('loss.value', value_loss)
        stats.mean('loss.policy', policy_loss)
        stats.mean('progress.resid-var', (target_value - d.v).pow(2).mean(), target_value.pow(2).mean())

        p0 = d0.prior.float().where(d0.prior > -np.inf, zeros)
        stats.mean('progress.kl-div.prior', (l0 - l).mul(l0.exp()).sum(-1).mean())
        stats.mean('progress.kl-div.target', (p0 - l).mul(p0.exp()).sum(-1).mean())

        stats.mean('rel-entropy.policy', *rel_entropy(d.logits)) 
        stats.mean('rel-entropy.targets', *rel_entropy(d0.logits))

        stats.mean('v.target.mean', target_value.mean())
        stats.mean('v.target.std', target_value.std())
        stats.mean('v.outputs.mean', d.v.mean())
        stats.mean('v.outputs.std', d.v.std())

        stats.rate('sample-rate.learner', t.terminal.nelement())
        stats.rate('step-rate.learner', 1)
        stats.cumsum('count.learner-steps', 1)
        # stats.rel_gradient_norm('rel-norm-grad', agent)

        stats.mean('opt.lr', np.mean([p['lr'] for p in opt.param_groups]))
        stats.mean('opt.step-std', (new - old).pow(2).mean().pow(.5))
        stats.max('opt.step-max', (new - old).abs().max())

def worldfunc(n_envs, device='cuda'):
    return hex.Hex.initial(n_envs=n_envs, boardsize=9, device=device)

def agentfunc(device='cuda'):
    worlds = worldfunc(n_envs=1, device=device)
    network = networks.FCModel(worlds.obs_space, worlds.action_space).to(worlds.device)
    return mcts.MCTSAgent(network, n_nodes=64)

def warm_start(agent, opt, parent):
    if parent:
        parent = runs.resolve(parent)
        sd = storage.load_latest(parent, device='cuda')
        agent.load_state_dict(sd['agent'])
        opt.load_state_dict(sd['opt'])
    return parent

def mix(worlds, T=2500):
    for _ in range(T):
        actions = torch.distributions.Categorical(probs=worlds.valid.float()).sample()
        worlds, transitions = worlds.step(actions)
    return worlds

@arrdict.mapping
def half(x):
    if isinstance(x, torch.Tensor) and x.dtype == torch.float:
        return x.half()
    else:
        return x

def run(device='cuda'):
    buffer_length = 16 
    batch_size = 16*1024
    n_envs = 16*1024
    buffer_inc = batch_size//n_envs

    worlds = mix(worldfunc(n_envs, device=device))
    agent = agentfunc(device)
    network = agent.network

    opt = torch.optim.Adam(network.parameters(), lr=1e-2, amsgrad=True)
    scaler = torch.cuda.amp.GradScaler()
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: min(e/100, 1))

    league = leagues.League(agent, agentfunc, worlds.n_envs, device=worlds.device)

    parent = warm_start(agent, opt, '')

    desc = 'back to Dirichlet'
    run = runs.new_run(boardsize=worlds.boardsize, parent=parent, description=desc)

    archive.archive(run)

    buffer = []
    idxs = learning.batch_indices(buffer_length, n_envs, batch_size, worlds.device)
    with logs.to_run(run), stats.to_run(run), \
            arena.monitor(run, worldfunc, agentfunc, device=worlds.device):
        while True:

            # Collect experience
            while len(buffer) < buffer_length:
                with torch.no_grad():
                    decisions = agent(worlds, value=True)
                new_worlds, transition = worlds.step(decisions.actions)

                buffer.append(arrdict.arrdict(
                    worlds=worlds,
                    decisions=decisions.half(),
                    transitions=half(transition),
                    is_prime=league.is_prime).detach())

                league.update(agent, worlds.seats, transition)

                worlds = new_worlds

                log.info('actor stepped')

            # Optimize
            chunk, buffer = as_chunk(buffer, buffer_inc)
            optimize(network, scaler, opt, chunk[next(idxs)])
            sched.step()
            
            log.info('learner stepped')

            sd = storage.state_dicts(agent=agent, opt=opt)
            storage.throttled_latest(run, sd, 60)
            storage.throttled_snapshot(run, sd, 900)
            storage.throttled_raw(run, 'model', lambda: pickle.dumps(network), 900)
            stats.gpu(worlds.device, 15)

@profiling.profilable
def benchmark_experience_collection(n_envs=8192, T=4):
    import pandas as pd

    if n_envs is None:
        ns = np.logspace(0, 15, 16, base=2, dtype=int)
        return pd.Series({n: benchmark_experience_collection(n) for n in ns})

    torch.manual_seed(0)
    worlds = worldfunc(n_envs)
    agent = agentfunc()

    agent(worlds) # warmup

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(T):
        decisions = agent(worlds)
        new_worlds, transition = worlds.step(decisions.actions)
        worlds = new_worlds
        print('actor stepped')
    torch.cuda.synchronize()
    rate = (T*n_envs)/(time.time() - start)
    print(f'{n_envs}: {rate}/sample')

    return rate

if __name__ == '__main__':
    with torch.autograd.profiler.emit_nvtx():
        benchmark_experience_collection()
