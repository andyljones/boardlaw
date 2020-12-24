import time
import numpy as np
import torch
from rebar import paths, widgets, logging, stats, arrdict, storing, timer
from .. import hex, mcts, networks, learning, validation, analysis, arena
from .common import worldfunc, agentfunc
from torch.nn import functional as F
from logging import getLogger
from itertools import cycle

log = getLogger(__name__)

@torch.no_grad()
def chunk_stats(chunk, n_new):
    with stats.defer():
        tail = chunk[-n_new:]
        d, t = tail.decisions, tail.transitions
        n_trajs = t.terminal.sum()
        n_inputs = t.terminal.size(0)
        n_samples = t.terminal.nelement()
        n_sims = d.n_sims.sum()
        stats.rate('sample-rate/actor', n_samples)
        stats.mean('traj-length', n_samples, n_trajs)
        stats.cumsum('count/traj', n_trajs)
        stats.cumsum('count/inputs', n_inputs)
        stats.cumsum('count/chunks', 1)
        stats.cumsum('count/samples', n_samples)
        stats.cumsum('count/sims', n_sims)
        stats.rate('step-rate/chunks', 1)
        stats.rate('step-rate/inputs', n_inputs)
        stats.rate('sim-rate', n_sims)
        stats.mean('mcts-n-leaves', d.n_leaves.float().mean())

        rewards = t.rewards.sum(0).sum(0)
        for i, r in enumerate(rewards):
            stats.mean(f'reward/seat-{i}', r, n_trajs)

        d, t = chunk.decisions, chunk.transitions
        v = d.v[t.terminal]
        r = t.rewards[t.terminal]
        stats.mean('progress/terminal-corr', ((v - v.mean())*(r - r.mean())).mean()/(v.var()*r.var())**.5)

        v = d.v[:-1][t.terminal[1:]]
        r = t.rewards[1:][t.terminal[1:]]
        stats.mean('progress/terminal-1-corr', ((v - v.mean())*(r - r.mean())).mean()/(v.var()*r.var())**.5)

def rel_entropy(logits, valid):
    zeros = torch.zeros_like(logits)
    logits = logits.where(valid, zeros)
    probs = logits.exp().where(valid, zeros)
    return (-(logits*probs).sum(-1).mean(), torch.log(valid.sum(-1).float()).mean())

def optimize(network, opt, batch):
    w, d0, t = batch.worlds, batch.decisions, batch.transitions
    d = network(w, value=True)

    zeros = torch.zeros_like(d.logits)
    policy_loss = -(d0.logits.exp()*d.logits).where(w.valid, zeros).sum(axis=-1).mean()

    terminal = torch.stack([t.terminal for _ in range(w.n_seats)], -1)
    target_value = learning.reward_to_go(t.rewards, d0.v, terminal, terminal, gamma=1)
    value_loss = (target_value - d.v).square().mean()
    
    loss = policy_loss + value_loss 
    
    opt.zero_grad()
    loss.backward()

    opt.step()

    with stats.defer():
        stats.mean('loss/value', value_loss)
        stats.mean('loss/policy', policy_loss)
        stats.mean('progress/resid-var', (target_value - d.v).pow(2).mean(), target_value.pow(2).mean())
        stats.mean('progress/kl-div', -(d0.logits - d.logits).where(w.valid, zeros).sum(-1).div(w.valid.float().sum(-1)).mean())

        stats.mean('rel-entropy/policy', *rel_entropy(d.logits, w.valid)) 
        stats.mean('rel-entropy/targets', *rel_entropy(d0.logits, w.valid))

        stats.mean('v-target/mean', target_value.mean())
        stats.mean('v-target/std', target_value.std())

        stats.rate('sample-rate/learner', t.terminal.nelement())
        stats.rate('step-rate/learner', 1)
        stats.cumsum('count/learner-steps', 1)
        # stats.rel_gradient_norm('rel-norm-grad', agent)

def run():
    buffer_length = 16 
    batch_size = 8192
    n_envs = 8192
    buffer_inc = batch_size//n_envs

    worlds = worldfunc(n_envs)
    agent = agentfunc()
    opt = torch.optim.Adam(agent.evaluator.parameters(), lr=1e-3, amsgrad=True)

    run_name = paths.timestamp('residual-9x9')
    paths.clear(run_name)
    with logging.to_dir(run_name), stats.to_dir(run_name), \
            arena.monitor(run_name, worldfunc, agentfunc):
        buffer = []
        idxs = cycle(learning.batch_indices(buffer_length, n_envs, batch_size, worlds.device))
        while True:

            while len(buffer) < buffer_length:
                decisions = agent(worlds, value=True)
                new_worlds, transition = worlds.step(decisions.actions)
                buffer.append(arrdict.arrdict(
                    worlds=worlds,
                    decisions=decisions,
                    transitions=transition).detach())
                worlds = new_worlds
                log.info('actor stepped')
                
            chunk = arrdict.stack(buffer)
            chunk_stats(chunk, buffer_inc)

            optimize(agent.evaluator, opt, chunk[:, next(idxs)])
            log.info('learner stepped')
            
            buffer = buffer[buffer_inc:]

            storing.store_latest(run_name, throttle=60, agent=agent, opt=opt)
            storing.store_periodic(run_name, throttle=900, agent=agent, opt=opt)
            stats.gpu.memory(worlds.device)
            stats.gpu.vitals(worlds.device, throttle=15)