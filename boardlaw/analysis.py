import time
import numpy as np
import torch
from rebar import arrdict, recording
from pavlov import runs, storage
from logging import getLogger
from . import arena

log = getLogger(__name__)

def combine_actions(decisions, masks):
    actions = torch.cat([d.actions for d in decisions.values()])
    for mask, decision in zip(masks.values(), decisions.values()):
        actions[mask] = decision.actions
    return actions

def expand(exemplar, n_envs):
    if exemplar.dtype in (torch.half, torch.float, torch.double):
        default = np.nan
    elif exemplar.dtype in (torch.short, torch.int, torch.long):
        default = -1
    else:
        raise ValueError('Don\'t have a default for "{exemplar.dtype}"')
    shape = (n_envs, *exemplar.shape[1:])
    return torch.full(shape, default, dtype=exemplar.dtype, device=exemplar.device)
    
def combine_decisions(dtrace, mtrace):
    agents = {a for d in dtrace for a in d}
    n_envs = next(iter(mtrace[0].values())).size(0)
    results = arrdict.arrdict()
    for a in agents:
        exemplar = [d[a] for d in dtrace if a in d][0]
        device = next(iter(arrdict.leaves(exemplar))).device

        a_results = []
        for d, m in zip(dtrace, mtrace):
            expanded = exemplar.map(expand, n_envs=n_envs)
            if a in m:
                expanded[m[a]] = d[a]
                expanded['mask'] = m[a]
            else:
                expanded['mask'] = torch.zeros((n_envs,), dtype=bool, device=device)
            a_results.append(expanded)
        results[str(a)] = arrdict.stack(a_results)
    return results

@torch.no_grad()
def rollout(worlds, agents, n_steps=None, n_trajs=None, n_reps=None, **kwargs):
    assert sum(x is not None for x in (n_steps, n_trajs, n_reps)) == 1, 'Must specify exactly one of n_steps or n_trajs or n_reps'

    trace, dtrace, mtrace = [], [], []
    steps, trajs = 0, 0
    reps = torch.zeros(worlds.n_envs, device=worlds.device)
    while True:
        decisions, masks = {}, {}
        for i, agent in enumerate(agents):
            mask = worlds.seats == i
            if mask.any():
                decisions[i] = agent(worlds[mask], **kwargs)
                masks[i] = mask

        actions = combine_actions(decisions, masks)
        
        worlds, transitions = worlds.step(actions)
        trace.append(arrdict.arrdict(
            actions=actions,
            transitions=transitions,
            worlds=worlds))
        
        mtrace.append(masks)
        dtrace.append(decisions)

        steps += 1

        if n_steps and (steps >= n_steps):
            break
        trajs += transitions.terminal.sum()
        if n_trajs and (trajs >= n_trajs):
            break
        reps += transitions.terminal
        if n_reps and (reps >= n_reps).all():
            break

    trace = arrdict.stack(trace)
    trace['decisions'] = combine_decisions(dtrace, mtrace)

    return trace

def plot_all(f):

    def proxy(state):
        import numpy as np
        import matplotlib.pyplot as plt

        B = state.seats.shape[0]
        assert B < 65, f'Plotting {B} traces will be prohibitively slow' 
        n_rows = int(B**.5)
        n_cols = int(np.ceil(B/n_rows))
        # Overlapping any more than this seems to distort the hexes. No clue why.
        fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False, gridspec_kw={'wspace': 0})

        for e in range(B):
            f(state, e, ax=axes.flatten()[e])
        
        return fig
    return proxy

def record_worlds(worlds, N=0):
    state = arrdict.numpyify(worlds)
    with recording.ParallelEncoder(plot_all(worlds.plot_worlds), N=N, fps=1) as encoder:
        for i in range(state.board.shape[0]):
            encoder(state[i])
    return encoder
    
def record(world, agents, N=0, **kwargs):
    trace = rollout(world, agents, **kwargs)
    return record_worlds(trace.worlds, N=N)