from . import hex, agents, learning, matchers
from rebar import arrdict, stats, widgets, logging, paths, plots, storing
import numpy as np
import torch
from logging import getLogger

log = getLogger(__name__)

def as_chunk(buffer):
    chunk = arrdict.stack(buffer)
    with stats.defer():
        i, r = chunk.inputs, chunk.responses
        stats.rate('sample-rate/actor', i.terminal.nelement())
        stats.mean('traj-length', i.terminal.nelement(), i.terminal.sum())
        stats.cumsum('count/traj', i.terminal.sum())
        stats.cumsum('count/inputs', i.terminal.size(0))
        stats.cumsum('count/chunks', 1)
        stats.cumsum('count/samples', i.terminal.nelement())
        stats.rate('step-rate/chunks', 1)
        stats.rate('step-rate/inputs', i.terminal.size(0))
        stats.mean('step-reward', r.reward.sum(), r.reward.nelement())
        stats.mean('traj-reward/mean', r.reward.sum(), i.terminal.sum())
        stats.mean('traj-reward/positive', r.reward.clamp(0, None).sum(), i.terminal.sum())
        stats.mean('traj-reward/negative', r.reward.clamp(None, 0).sum(), i.terminal.sum())
    return chunk

def optimize(agent, opt, batch, entropy=1e-2, gamma=.99, clip=.2):
    #TODO: Env should emit batch data delayed so that it can fix the terminal/reward itself.
    deinterlaced = matchers.deinterlace(matchers.symmetrize(batch))

    i, d0, r = deinterlaced.inputs, deinterlaced.decisions, deinterlaced.responses
    d = agent(i, value=True)

    logits = learning.flatten(d.logits)
    old_logits = learning.flatten(learning.gather(d0.logits, d0.actions))
    new_logits = learning.flatten(learning.gather(d.logits, d0.actions))
    ratio = (new_logits - old_logits).exp().clamp(.05, 20)

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

def train():
    """ 
    Player chunks:
     * Adapt present_value, v_trace to track the player
     * Adapt recurrent state to ... what? How'd you deal with multi-agent experience collection?
    """
    buffer_size = 64
    n_envs = 2048
    batch_size = 8*1024

    env = hex.Hex(n_envs)
    agent = agents.Agent(env.obs_space, env.action_space).to(env.device)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4, amsgrad=True)

    run_name = paths.timestamp('test')
    compositor = widgets.Compositor()
    paths.clear(run_name)
    with logging.via_dir(run_name, compositor), stats.via_dir(run_name, compositor):
        inputs = env.reset()
        while True:
            buffer = []
            for _ in range(buffer_size):
                decisions = agent(inputs[None], sample=True, value=True).squeeze(0)
                responses, new_inputs = env.step(decisions.actions)
                buffer.append(arrdict.arrdict(
                    inputs=inputs,
                    decisions=decisions,
                    responses=responses).detach())
                inputs = new_inputs.detach()
                
            chunk = as_chunk(buffer)

            for idxs in learning.batch_indices(chunk, batch_size):
                kl = optimize(agent, opt, chunk[:, idxs])

                log.info(f'learner stepped')
                if kl > .02:
                    log.info('kl div exceeded')
                    break

            storing.store_latest(run_name, {'agent': agent, 'opt': opt}, throttle=60)
            storing.store_periodic(run_name, {'agent': agent, 'opt': opt}, throttle=600)

def compare(run_name=-1, left=-2, right=-1):
    env = hex.Hex(256)
    agent = [agents.Agent(env.obs_space, env.action_space).to(env.device) for _ in range(2)]
    agent[0].load_state_dict(storing.load_periodic(run_name, idx=left)['agent'])
    agent[1].load_state_dict(storing.load_periodic(run_name, idx=right)['agent'])
    return matchers.winrate(env, agent)

def compare_all(run_name=-1):
    import pandas as pd
    from itertools import combinations

    df = storing.stored_periodic(run_name)
    rates = {(l, r): compare(-1, l, r) for l, r in combinations(range(len(df)), 2)}
    rates = pd.DataFrame.from_dict(rates).iloc[1].unstack(1)
    
    return rates

def ratings(run_name=-1):
    import cvxpy as cp
    import pandas as pd

    rates = compare_all(run_name)

    names = list(set(rates.index) | set(rates.columns))
    full = rates.reindex(index=names, columns=names)
    comp = full.combine_first(1 - full.T).fillna(.5)

    targets = np.log2(1/comp.values - 1).clip(-10, +10)

    #TODO: Is this even remotely how to calculate Elos?
    r = cp.Variable(len(names))
    loss = cp.sum_squares(r[:, None] - r[None, :] - targets)
    cp.Problem(cp.Minimize(loss), [r[0] == 0]).solve()
    return pd.Series(r.value, names)