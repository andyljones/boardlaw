import numpy as np
import torch
from rebar import paths, widgets, logging, stats, arrdict, storing
from . import hex, mcts, networks, learning
from torch.nn import functional as F
from logging import getLogger

log = getLogger(__name__)

def as_chunk(buffer):
    chunk = arrdict.stack(buffer)
    with stats.defer():
        i, r = chunk.inputs, chunk.responses
        n_trajs = r.terminal.sum()
        n_inputs = r.terminal.size(0)
        n_samples = r.terminal.nelement()
        stats.rate('sample-rate/actor', n_samples)
        stats.mean('traj-length', n_samples, n_trajs)
        stats.cumsum('count/traj', n_trajs)
        stats.cumsum('count/inputs', n_inputs)
        stats.cumsum('count/chunks', 1)
        stats.cumsum('count/samples', n_samples)
        stats.rate('step-rate/chunks', 1)
        stats.rate('step-rate/inputs', n_inputs)
        stats.mean('step-reward', r.rewards.sum(), n_samples)
        stats.mean('traj-reward/mean', r.rewards.sum(), n_trajs)
        stats.mean('traj-reward/positive', r.rewards.clamp(0, None).sum(), n_trajs)
        stats.mean('traj-reward/negative', r.rewards.clamp(None, 0).sum(), n_trajs)
    return chunk

def optimize(network, opt, batch):
    #TODO: Env should emit batch data delayed so that it can fix the terminal/reward itself.
    i, d0, r = batch.inputs, batch.decisions, batch.responses
    d = network(i, value=True)

    target_logits = d0.logits
    actual_probs = d.logits.exp()
    policy_terms = -(actual_probs*target_logits).where(batch.inputs.valid, torch.zeros_like(actual_probs))
    policy_loss = policy_terms.sum(axis=1).mean()

    terminal = torch.stack([r.terminal, r.terminal], -1)
    target_value = learning.reward_to_go(r.rewards, d0.v, terminal, terminal, gamma=1)
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

        stats.rate('sample-rate/learner', r.terminal.nelement())
        stats.rate('step-rate/learner', 1)
        stats.cumsum('count/learner-steps', 1)
        # stats.rel_gradient_norm('rel-norm-grad', agent)

def run():
    env = hex.Hex(n_envs=512, boardsize=5, device='cuda')
    network = networks.Network(env.obs_space, env.action_space, width=128).to(env.device)
    agent = mcts.MCTSAgent(env, network, n_nodes=16)
    opt = torch.optim.Adam(network.parameters(), lr=3e-4, amsgrad=True)

    run_name = 'az-test'
    compositor = widgets.Compositor()
    paths.clear(run_name)
    with logging.via_dir(run_name, compositor), stats.via_dir(run_name, compositor):
        inputs = env.reset()

        while True:
            buffer = []
            for _ in range(32):
                decisions = agent(inputs, responses, value=True)
                responses, new_inputs = env.step(decisions.actions)
                buffer.append(arrdict.arrdict(
                    inputs=inputs,
                    decisions=decisions,
                    responses=responses).detach())
                inputs = new_inputs.detach()
                
            chunk = as_chunk(buffer)

            for idxs in learning.batch_indices(chunk, 2048):
                optimize(network, opt, chunk[:, idxs])
                log.info('learner stepped')

            storing.store_latest(run_name, {'network': network, 'opt': opt}, throttle=60)
            storing.store_periodic(run_name, {'network': network, 'opt': opt}, throttle=600)
