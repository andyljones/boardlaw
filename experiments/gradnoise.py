import numpy as np
import torch
from boardlaw.main import agentfunc, worldfunc, mix, half, as_chunk
from pavlov import storage
from rebar import arrdict
from logging import getLogger

log = getLogger(__name__)

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

    opt.zero_grad()
    scaler.scale(loss).backward()
    old = {k: v.detach().clone() for k, v in network.state_dict().items()}
    scaler.step(opt)
    scaler.update()
    network.load_state_dict(old)

def gradients(run, i, n_envs=16*1024, buffer_len=64, device='cuda'):

    #TODO: Restore league and sched when you go back to large boards
    worlds = mix(worldfunc(n_envs, device=device))
    agent = agentfunc(device)
    network = agent.network

    opt = torch.optim.Adam(network.parameters(), lr=0., amsgrad=True)
    scaler = torch.cuda.amp.GradScaler()

    sd = storage.load_snapshot(run, i)
    agent.load_state_dict(sd['agent'])
    opt.load_state_dict(sd['opt'])

    buffer = []

    idxs = (torch.randint(buffer_len, (n_envs,), device=device), torch.arange(n_envs, device=device))
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

        yield torch.cat([p.grad.flatten() for p in network.parameters() if p.grad is not None])

def official_way(gs, Bsmall):
    Gbig2 = gs.mean(0).pow(2).mean()
    Gsmall2 = gs.pow(2).mean(1).mean()

    Bbig = gs.size(0)*Bsmall

    G2 = 1/(Bbig - Bsmall)*(Bbig*Gbig2 - Bsmall*Gsmall2)
    S = 1/(1/Bsmall - 1/Bbig)*(Gsmall2 - Gbig2)

    return S/G2

def sensible_way(gs, Bsmall):
    return Bsmall*(gs - gs.mean(0, keepdims=True)).pow(2).mean()/gs.mean(0).pow(2).mean()

def adam_way(run, i, Bsmall):
    # Doesn't work. Or at least it's wayyyy off from the 'official' estimates
    # Gets S pretty much right, but is 5x too small on G2.
    sd = storage.load_snapshot(run, i)['opt']['state']
    m0 = torch.cat([s['exp_avg'].flatten() for s in sd.values()])
    v0 = torch.cat([s['exp_avg_sq'].flatten() for s in sd.values()])

    S = Bsmall*(v0.mean() - m0.pow(2).mean())
    G2 = m0.pow(2).mean()

    return S/G2

def run():
    gs = []
    B = 16*1024
    for _, g in zip(range(128), gradients('*large-model', B)):
        log.info(f'{len(gs)} gradients')
        gs.append(g)
    gs = torch.stack(gs).cpu()

    Bnoise = official_way(gs, B)