from . import hex, agents, learning
from rebar import arrdict
import torch
from logging import getLogger

log = getLogger(__name__)

def optimize(agent, opt, batch, entropy=1e-2, gamma=.99, clip=.2):
    i, d0, r = batch.inputs, batch.decisions, batch.responses
    d = agent(i, value=True)

    logits = learning.flatten(d.logits)
    old_logits = learning.flatten(learning.gather(d0.logits, d0.actions))
    new_logits = learning.flatten(learning.gather(d.logits, d0.actions))
    ratio = (new_logits - old_logits).exp().clamp(.05, 20)

    v_target = learning.v_trace(ratio, d.value, r.reward, i.reset, gamma=gamma)
    v_clipped = d0.value + torch.clamp(d.value - d0.value, -10, +10)
    v_loss = .5*torch.max((d.value - v_target)**2, (v_clipped - v_target)**2).mean()

    adv = learning.generalized_advantages(d.value, r.reward, d.value, i.reset, gamma=gamma)
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

    return kl_div

def train():
    buffer_size = 64
    n_envs = 1024
    batch_size = 8*1024

    env = hex.Hex(n_envs)
    agent = agents.Agent(env.obs_space, env.action_space).to(env.device)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4, amsgrad=True)

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
            
        chunk = arrdict.stack(buffer)

        for idxs in learning.batch_indices(chunk, batch_size):
            kl = optimize(agent, opt, chunk[:, idxs])

            log.info(f'learner stepped')
            if kl > .02:
                log.info('kl div exceeded')
                break