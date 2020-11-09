import torch
from rebar import paths, widgets, logging, stats, arrdict
from . import hex, mcts, networks, learning
from torch.nn import functional as F

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

def optimize(network, opt, batch, entropy=1e-2, gamma=.99, clip=.2):
    #TODO: Env should emit batch data delayed so that it can fix the terminal/reward itself.
    i, d0, r = batch.inputs, batch.decisions, batch.responses
    d = network(i, value=True)

    target_logits = d0.logits[batch.inputs.valid]
    actual_logits = d.logits[batch.inputs.valid]
    F.nll_loss()


    v_target = learning.reward_to_go(r.reward, d0.value, i.reset, i.terminal, gamma=gamma)
    v_clipped = d0.value + torch.clamp(d.value - d0.value, -10, +10)
    v_loss = .5*torch.max((d.value - v_target)**2, (v_clipped - v_target)**2).mean()

    adv = learning.generalized_advantages(d.value, r.reward, d.value, i.reset, i.terminal, gamma=gamma)
    normed_adv = (adv - adv.mean())/(1e-3 + adv.std())
    free_adv = ratio*normed_adv
    clip_adv = torch.clamp(ratio, 1-clip, 1+clip)*normed_adv
    p_loss = -torch.min(free_adv, clip_adv).mean()

    h_loss = (logits.exp()*logits).sum(-1).mean()
    loss = v_loss + p_loss + entropy*h_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 100.)
    torch.nn.utils.clip_grad_norm_(agent.value.parameters(), 100.)

    opt.step()

    kl_div = -(new_logits - old_logits).mean().detach()

    with stats.defer():
        stats.mean('loss/value', v_loss)
        stats.mean('loss/policy', p_loss)
        stats.mean('loss/entropy', h_loss)
        stats.mean('resid-var/v', (v_target - d.value).pow(2).mean(), v_target.pow(2).mean())
        stats.mean('rel-entropy', -(logits.exp()*logits).sum(-1).mean()/np.log(logits.shape[-1]))
        stats.mean('kl-div', kl_div) 

        stats.mean('v-target/mean', v_target.mean())
        stats.mean('v-target/std', v_target.std())

        stats.mean('adv/z-mean', adv.mean())
        stats.mean('adv/z-std', adv.std())
        stats.max('adv/z-max', adv.abs().max())

        stats.rate('sample-rate/learner', i.reset.nelement())
        stats.rate('step-rate/learner', 1)
        stats.cumsum('count/learner-steps', 1)
        # stats.rel_gradient_norm('rel-norm-grad', agent)

        stats.mean('param/gamma', gamma)
        stats.mean('param/entropy', entropy)

    return kl_div

def run():
    env = hex.Hex(n_envs=8, boardsize=5, device='cpu')
    network = networks.Network(env.obs_space, env.action_space, width=16)
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
                decisions = agent(inputs, value=True)
                responses, new_inputs = env.step(decisions.actions)
                buffer.append(arrdict.arrdict(
                    inputs=inputs,
                    decisions=decisions,
                    responses=responses).detach())
                inputs = new_inputs.detach()
                
            chunk = as_chunk(buffer)

            for idxs in learning.batch_indices(chunk, 128):
                optimize(network, opt, chunk[:, idxs])
