import numpy as np
import torch
from rebar import paths, widgets, logging, stats, arrdict, storing
from . import hex, mcts, networks, learning, validation
from torch.nn import functional as F
from logging import getLogger
from itertools import cycle

log = getLogger(__name__)

def system_stats():
    with stats.defer():
        torch.cuda.reset_peak_memory_stats()
        stats.mean('system/gpu-alloc', torch.cuda.max_memory_allocated()/1e6)
        stats.mean('system/gpu-cached', torch.cuda.max_memory_cached()/1e6)

def chunk_stats(chunk):
    with stats.defer():
        t = chunk.transition
        n_trajs = t.terminal.sum()
        n_inputs = t.terminal.size(0)
        n_samples = t.terminal.nelement()
        stats.rate('sample-rate/actor', n_samples)
        stats.mean('traj-length', n_samples, n_trajs)
        stats.cumsum('count/traj', n_trajs)
        stats.cumsum('count/inputs', n_inputs)
        stats.cumsum('count/chunks', 1)
        stats.cumsum('count/samples', n_samples)
        stats.cumsum('count/sims', chunk.decisions.n_sims.sum())
        stats.rate('step-rate/chunks', 1)
        stats.rate('step-rate/inputs', n_inputs)
        stats.mean('mcts-branching', chunk.decisions.branching.mean())

        rewards = chunk.transition.rewards.sum(0).sum(0)
        for i, r in enumerate(rewards):
            stats.mean(f'reward/seat-{i}', r, n_trajs)
    return chunk

def optimize(network, opt, batch):
    w, d0, t = batch.world, batch.decisions, batch.transition
    d = network(w, value=True)

    target_logits = d0.logits
    target_logits[torch.isinf(target_logits)] = 0.
    actual_probs = d.logits.exp()
    policy_loss = -(actual_probs*target_logits).sum(axis=1).mean()

    terminal = torch.stack([t.terminal for _ in range(w.n_seats)], -1)
    target_value = learning.reward_to_go(t.rewards, d0.v, terminal, terminal, gamma=1)
    value_loss = (target_value - d.v).square().mean()
    
    loss = policy_loss + value_loss 
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.policy.parameters(), 100.)
    torch.nn.utils.clip_grad_norm_(network.value.parameters(), 100.)

    opt.step()

    with stats.defer():
        stats.mean('loss/value', value_loss)
        stats.mean('loss/policy', policy_loss)
        stats.mean('resid-var/v', (target_value - d.v).pow(2).mean(), target_value.pow(2).mean())
        stats.mean('rel-entropy', -(d.logits.exp()*d.logits).sum(-1).mean()/np.log(d.logits.size(-1)))

        stats.mean('v-target/mean', target_value.mean())
        stats.mean('v-target/std', target_value.std())

        stats.rate('sample-rate/learner', t.terminal.nelement())
        stats.rate('step-rate/learner', 1)
        stats.cumsum('count/learner-steps', 1)
        # stats.rel_gradient_norm('rel-norm-grad', agent)

def run():
    buffer_length = 8
    batch_size = 1024
    n_envs = 1024
    buffer_inc = batch_size//n_envs

    # world = hex.Hex.initial(n_envs=n_envs, boardsize=5, device='cuda')
    world = hex.Hex.initial(n_envs=n_envs, boardsize=5, device='cuda')
    network = networks.Network(world.obs_space, world.action_space, width=128).to(world.device)
    agent = mcts.MCTSAgent(network, n_nodes=16)
    opt = torch.optim.Adam(network.parameters(), lr=1e-3, amsgrad=True)

    run_name = paths.timestamp('az-test')
    compositor = widgets.Compositor()
    paths.clear(run_name)
    with logging.via_dir(run_name, compositor), stats.via_dir(run_name, compositor):
        buffer = []
        idxs = cycle(learning.batch_indices(buffer_length, n_envs, batch_size))
        while True:
            while len(buffer) < buffer_length:
                decisions = agent(world, value=True)
                new_world, transition = world.step(decisions.actions)
                buffer.append(arrdict.arrdict(
                    world=world,
                    decisions=decisions,
                    transition=transition).detach())
                world = new_world
                log.info('actor stepped')
                
            chunk = arrdict.stack(buffer)
            chunk_stats(chunk[-buffer_inc:])

            optimize(network, opt, chunk[:, next(idxs)])
            log.info('learner stepped')
            
            buffer = buffer[buffer_inc:]

            storing.store_latest(run_name, {'network': network, 'opt': opt}, throttle=60)
            storing.store_periodic(run_name, {'network': network, 'opt': opt}, throttle=600)
            system_stats()