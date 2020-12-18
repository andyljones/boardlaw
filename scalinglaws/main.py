import time
import numpy as np
import torch
from rebar import paths, widgets, logging, stats, arrdict, storing, timer
from . import hex, mcts, networks, learning, validation, analysis, arena, buffering
from torch.nn import functional as F
from logging import getLogger

log = getLogger(__name__)

@torch.no_grad()
def actor_stats(sample):
    with stats.defer():
        d, t = sample.decisions, sample.transitions
        n_trajs = t.terminal.sum()
        n_samples = t.terminal.size(0)
        n_sims = d.n_sims.sum()
        stats.rate('sample-rate/actor', n_samples)
        stats.mean('traj-length', n_samples, n_trajs)
        stats.cumsum('count/traj', n_trajs)
        stats.cumsum('count/inputs', 1)
        stats.cumsum('count/chunks', 1)
        stats.cumsum('count/samples', n_samples)
        stats.cumsum('count/sims', n_sims)
        stats.rate('step-rate/chunks', 1)
        stats.rate('step-rate/inputs', 1)
        stats.rate('sim-rate', n_sims)
        stats.mean('mcts-n-leaves', d.n_leaves.float().mean())

        rewards = t.rewards.sum(0)
        for i, r in enumerate(rewards):
            stats.mean(f'reward/seat-{i}', r, n_trajs)

        v = d.v[t.terminal]
        r = t.rewards[t.terminal]
        stats.mean('progress/terminal-corr', ((v - v.mean())*(r - r.mean())).mean()/(v.var()*r.var())**.5)

        # v = d.v[:-1][t.terminal[1:]]
        # r = t.rewards[1:][t.terminal[1:]]
        # stats.mean('progress/terminal-1-corr', ((v - v.mean())*(r - r.mean())).mean()/(v.var()*r.var())**.5)

def rel_entropy(logits, valid):
    zeros = torch.zeros_like(logits)
    logits = logits.where(valid, zeros)
    probs = logits.exp().where(valid, zeros)
    return (-(logits*probs).sum(-1).mean(), torch.log(valid.sum(-1).float()).mean())

def optimize(network, opt, batch):
    d = network(batch, value=True)

    zeros = torch.zeros_like(d.logits)
    policy_loss = -(batch.logits.exp()*d.logits).where(batch.valid, zeros).sum(axis=-1).mean()

    value_loss = (batch.targets - d.v).square().mean()
    
    loss = policy_loss + value_loss 
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.policy.parameters(), 100.)
    torch.nn.utils.clip_grad_norm_(network.value.parameters(), 100.)

    opt.step()

    with stats.defer():
        stats.mean('loss/value', value_loss)
        stats.mean('loss/policy', policy_loss)
        stats.mean('progress/resid-var', (batch.targets - d.v).pow(2).mean(), batch.targets.pow(2).mean())
        stats.mean('progress/kl-div', -(batch.logits - d.logits).where(batch.valid, zeros).mean())

        stats.mean('rel-entropy/policy', *rel_entropy(d.logits, batch.valid)) 
        stats.mean('rel-entropy/targets', *rel_entropy(batch.logits, batch.valid))

        stats.mean('v-target/mean', batch.targets.mean())
        stats.mean('v-target/std', batch.targets.std())

        stats.rate('sample-rate/learner', batch.targets.size(0))
        stats.rate('step-rate/learner', 1)
        stats.cumsum('count/learner-steps', 1)
        # stats.rel_gradient_norm('rel-norm-grad', agent)

def worldfunc(n_envs, device='cuda'):
    return hex.Hex.initial(n_envs=n_envs, boardsize=11, device=device)

def agentfunc(device='cuda'):
    worlds = worldfunc(n_envs=1, device=device)
    network = networks.Network(worlds.obs_space, worlds.action_space, width=128).to(worlds.device)
    # network.trace(worlds)
    return mcts.MCTSAgent(network, n_nodes=64)

def run():
    batch_size = 1024
    n_envs = 1024

    worlds = worldfunc(n_envs)
    agent = agentfunc()
    opt = torch.optim.Adam(agent.evaluator.parameters(), lr=1e-3, amsgrad=True)
    buffer = buffering.Buffer(1024*1024//n_envs, keep=1.)

    run_name = paths.timestamp('az-test')
    paths.clear(run_name)
    with logging.to_dir(run_name), stats.to_dir(run_name):
        while True:
            decisions = agent(worlds, value=True)
            new_worlds, transition = worlds.step(decisions.actions)
            sample = arrdict.arrdict(
                worlds=worlds,
                decisions=decisions,
                transitions=transition).detach()
            buffer.add(sample)
            actor_stats(sample)
            worlds = new_worlds
            log.info('actor stepped')
                
            if not buffer.ready():
                log.info('Buffer not yet ready')
            else:
                batch = buffer.sample(batch_size)
                optimize(agent.evaluator, opt, batch)
                log.info('learner stepped')

            storing.store_latest(run_name, throttle=60, agent=agent, opt=opt)
            storing.store_periodic(run_name, throttle=900, agent=agent, opt=opt)
            stats.gpu.memory(worlds.device)
            stats.gpu.vitals(worlds.device, throttle=15)

def monitor(run_name=-1):
    compositor = widgets.Compositor()
    with logging.from_dir(run_name, compositor), stats.from_dir(run_name, compositor), \
            arena.monitor(run_name, worldfunc, agentfunc):
        while True:
            time.sleep(1)

def demo(run_name=-1):
    from scalinglaws import mohex

    n_envs = 4
    world = worldfunc(n_envs, device='cuda:1')
    agent = agentfunc(device='cuda:1')
    agent.load_state_dict(storing.select(storing.load_latest(run_name), 'agent'))
    mhx = mohex.MoHexAgent(presearch=False, max_games=1)
    analysis.record(world, [agent, agent], n_reps=1, N=0).notebook()

def compare(fst_run=-1, snd_run=-1, n_envs=256, device='cuda:1'):
    import pandas as pd

    world = worldfunc(n_envs, device=device)

    fst = agentfunc(device=device)
    fst.load_state_dict(storing.select(storing.load_latest(fst_run), 'agent'))

    snd = agentfunc(device=device)
    snd.load_state_dict(storing.select(storing.load_latest(snd_run), 'agent'))

    bw = analysis.rollout(world, [fst, snd], n_reps=1)
    bw_wins = (bw.transitions.rewards[bw.transitions.terminal.cumsum(0) <= 1] == 1).sum(0)

    wb = analysis.rollout(world, [snd, fst], n_reps=1)
    wb_wins = (wb.transitions.rewards[wb.transitions.terminal.cumsum(0) <= 1] == 1).sum(0)

    # Rows: black, white; cols: old, new
    wins = torch.stack([bw_wins, wb_wins.flipud()]).detach().cpu().numpy()

    return pd.DataFrame(wins/n_envs, ['black', 'white'], ['fst', 'snd'])

def benchmark_experience_collection():
    # Make sure to init cuda before running this 

    torch.manual_seed(0)
    n_envs = 4096
    worlds = worldfunc(n_envs)
    agent = agentfunc()

    with timer.timer(cuda=True) as t:
        for _ in range(16):
            decisions = agent(worlds, value=True)
            new_worlds, transition = worlds.step(decisions.actions)
            worlds = new_worlds
            log.info('actor stepped')
    print(f'{t/(16*n_envs)}/sample')