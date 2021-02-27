import time
import numpy as np
import torch
from rebar import arrdict, profiling, pickle
from pavlov import stats, logs, runs, storage, archive
from . import hex, mcts, networks, learning, validation, analysis, arena, storage
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
        stats.mean('corr.terminal', ((v - v.mean())*(w - w.mean())).mean()/(v.var()*w.var())**.5)

        v = d.v[:-1][t.terminal[1:]]
        w = t.rewards[1:][t.terminal[1:]]
        stats.mean('corr.penultimate', ((v - v.mean())*(w - w.mean())).mean()/(v.var()*w.var())**.5)

def as_chunk(buffer, batch_size):
    chunk = arrdict.stack(buffer)
    terminal = torch.stack([chunk.transitions.terminal for _ in range(chunk.worlds.n_seats)], -1)
    chunk['reward_to_go'] = learning.reward_to_go(
        chunk.transitions.rewards.float(), 
        chunk.decisions.v.float(), 
        terminal).half()

    n_new = batch_size//terminal.size(1)
    chunk_stats(chunk, n_new)
            
    buffer = buffer[n_new:]

    return chunk, buffer

def rel_entropy(logits):
    valid = (logits > -np.inf)
    zeros = torch.zeros_like(logits)
    logits = logits.where(valid, zeros)
    probs = logits.exp().where(valid, zeros)
    return (-(logits*probs).sum(-1).mean(), torch.log(valid.sum(-1).float()).mean())

def noise_scale(B, opt):
    step = list(opt.state.values())[0]['step']
    beta1, beta2 = opt.param_groups[0]['betas']
    m_bias = 1 - beta1**step
    v_bias = 1 - beta2**step

    m = 1/m_bias*torch.cat([s['exp_avg'].flatten() for s in opt.state.values() if 'exp_avg' in s])
    v = 1/v_bias*torch.cat([s['exp_avg_sq'].flatten() for s in opt.state.values() if 'exp_avg_sq' in s])

    # Follows from chasing the var through the defn of m
    inflator = (1 - beta1**2)/(1 - beta1)**2

    S = B*(v.mean() - m.pow(2).mean())
    G2 = inflator*m.pow(2).mean()

    return S/G2

def optimize(network, scaler, opt, batch):

    with torch.cuda.amp.autocast():
        d0 = batch.decisions
        d = network(batch.worlds)

        zeros = torch.zeros_like(d.logits)
        l = d.logits.where(d.logits > -np.inf, zeros)
        l0 = d0.logits.float().where(d0.logits > -np.inf, zeros)

        policy_loss = -(l0.exp()*l).sum(axis=-1).mean()

        target_value = batch.reward_to_go
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
        stats.mean('corr.resid-var', (target_value - d.v).pow(2).mean(), target_value.pow(2).mean())

        p0 = d0.prior.float().where(d0.prior > -np.inf, zeros)
        stats.mean('kl-div.behaviour', (p0 - l0).mul(p0.exp()).sum(-1).mean())
        stats.mean('kl-div.prior', (p0 - l).mul(p0.exp()).sum(-1).mean())

        stats.mean('rel-entropy.policy', *rel_entropy(d.logits)) 
        stats.mean('rel-entropy.targets', *rel_entropy(d0.logits))

        stats.mean('v.target.mean', target_value.mean())
        stats.mean('v.target.std', target_value.std())
        stats.mean('v.target.max', target_value.abs().max())
        stats.mean('v.outputs.mean', d.v.mean())
        stats.mean('v.outputs.std', d.v.std())
        stats.mean('v.outputs.max', d.v.abs().max())

        stats.mean('p.target.mean', l0.mean())
        stats.mean('p.target.std', l0.std())
        stats.mean('p.target.max', l0.abs().max())
        stats.mean('p.outputs.mean', l.mean())
        stats.mean('p.outputs.std', l.std())
        stats.mean('p.outputs.max', l.abs().max())

        stats.mean('policy-conc', l0.exp().max(-1).values.mean())

        stats.rate('sample-rate.learner', batch.transitions.terminal.nelement())
        stats.rate('step-rate.learner', 1)
        stats.cumsum('count.learner-steps', 1)
        # stats.rel_gradient_norm('rel-norm-grad', agent)

        stats.mean('step.std', (new - old).pow(2).mean().pow(.5))
        stats.max('step.max', (new - old).abs().max())

        grad = torch.cat([p.grad.flatten() for p in network.parameters() if p.grad is not None])
        stats.max('grad.max', grad.abs().max())
        stats.max('grad.std', grad.pow(2).mean().pow(.5))
        stats.max('grad.norm', grad.pow(2).sum().pow(.5))
        
        B = batch.transitions.terminal.nelement()
        stats.mean('noise-scale', noise_scale(B, opt))

def agent_factory(worldfunc, **kwargs):
    worlds = worldfunc(n_envs=1)
    network = networks.FCModel(worlds.obs_space, worlds.action_space, **kwargs).to(worlds.device)
    return mcts.MCTSAgent(network)

def warm_start(agent, opt, scaler, parent):
    if parent:
        parent = runs.resolve(parent)
        sd = storage.load_latest(parent, device='cuda')
        agent.load_state_dict(sd['agent'])
        opt.load_state_dict(sd['opt'])
        scaler.load_state_dict(sd['scaler'])
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

def run(boardsize, width, depth, nodes, desc):
    buffer_len = 64
    n_envs = 32*1024

    #TODO: Restore league and sched when you go back to large boards
    worlds = mix(hex.Hex.initial(n_envs, boardsize))
    network = networks.FCModel(worlds.obs_space, worlds.action_space, width=width, depth=depth).to(worlds.device)
    agent = mcts.MCTSAgent(network, n_nodes=nodes)

    opt = torch.optim.Adam(network.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    parent = warm_start(agent, opt, scaler, '')

    run = runs.new_run(
            description=desc, 
            params=dict(boardsize=worlds.boardsize, width=width, depth=depth, nodes=nodes, parent=parent))

    archive.archive(run)

    storer = storage.LogarithmicStorer(run, agent)

    buffer = []
    with logs.to_run(run), stats.to_run(run), \
            arena.mohex.run(run):
        #TODO: Upgrade this to handle batches that are some multiple of the env count
        idxs = (torch.randint(buffer_len, (n_envs,), device='cuda'), torch.arange(n_envs, device='cuda'))
        while True:

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

            stats.gpu(worlds.device, 15)

            finish = storer.step(agent, len(idxs[0]))
            if finish:
                break

        log.info('Finished')

def run_jittens():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['JITTENS_GPU']
    print(f'Devices set to "{os.environ["CUDA_VISIBLE_DEVICES"]}"')

    import ast
    d = ast.literal_eval(os.environ['JITTENS_PARAMS'])
    run(**d)
     