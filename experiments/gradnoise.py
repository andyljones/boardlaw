import pandas as pd
import numpy as np
import torch
from boardlaw.main import mix, half, as_chunk
from pavlov import storage
from rebar import arrdict
from logging import getLogger
from boardlaw.hex import Hex
from boardlaw.mcts import MCTSAgent

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
    worlds = mix(Hex.initial(n_envs, device=device))
    network = storage.load_raw(run, 'model')
    agent = MCTSAgent(network)

    opt = torch.optim.Adam(network.parameters(), lr=0., amsgrad=True)
    scaler = torch.cuda.amp.GradScaler()

    sd = storage.load_snapshot(run, i)
    agent.load_state_dict(sd['agent'])
    opt.load_state_dict(sd['opt'])
    scaler.load_state_dict(sd['scaler'])

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

    return (S/G2).item()

def sensible_way(gs, Bsmall):
    S = Bsmall*(gs - gs.mean(0, keepdims=True)).pow(2).mean()
    G2 = gs.mean(0).pow(2).mean()
    return (S/G2).item()

def adam_way(run, i, Bsmall):
    # Doesn't work. Or at least it's wayyyy off from the 'official' estimates
    # Gets S pretty much right, but is 5x too small on G2.
    sd = storage.load_snapshot(run, i)
    beta1, beta2 = sd['opt']['param_groups'][0]['betas']

    opt = sd['opt']['state']
    m0 = torch.cat([s['exp_avg'].flatten() for s in opt.values()])
    v0 = torch.cat([s['exp_avg_sq'].flatten() for s in opt.values()])

    S = (1 - beta1)*Bsmall*(v0.mean() - m0.pow(2).mean())
    G2 = m0.pow(2).mean()

    return (S/G2).item()

def run():
    run, idx = '*that-man', -1
    B = 16*1024

    gs = []
    for _, g in zip(range(32), gradients(run, idx)):
        log.info(f'{len(gs)} gradients')
        gs.append(g)
    gs = torch.stack(gs).cpu()

    # official    27668.0
    # sensible    25460.0
    # adam        31993.0
    # dtype: float64
    sizes = pd.Series(dict(
        official=official_way(gs, B),
        sensible=sensible_way(gs, B),
        adam=adam_way(run, -1, B)))