import time
import numpy as np
import torch
from rebar import arrdict, stats, recording
from logging import getLogger

log = getLogger(__name__)

def rollout(worlds, agents, n_steps=None, n_trajs=None, n_reps=None):
    assert sum(x is not None for x in (n_steps, n_trajs, n_reps)) == 1, 'Must specify exactly one of n_steps or n_trajs or n_reps'

    trace = []
    steps, trajs = 0, 0
    reps = torch.zeros(worlds.n_envs, device=worlds.device)
    while True:
        actions = torch.full((worlds.n_envs,), -1, device=worlds.device)
        for i, agent in enumerate(agents):
            mask = worlds.seats == i
            if mask.any():
                actions[mask] = agent(worlds[mask]).actions
        worlds, transitions = worlds.step(actions)
        trace.append(arrdict.arrdict(
            actions=actions,
            transitions=transitions,
            worlds=worlds))
        steps += 1
        if n_steps and (steps >= n_steps):
            break
        trajs += transitions.terminal.sum()
        if n_trajs and (trajs >= n_trajs):
            break
        reps += transitions.terminal
        if n_reps and (reps >= n_reps).all():
            break
    return arrdict.stack(trace)

def plot_all(f):

    def proxy(state):
        import numpy as np
        import matplotlib.pyplot as plt

        B = state.seat.shape[0]
        assert B < 65, f'Plotting {B} traces will be prohibitively slow' 
        n_rows = int(B**.5)
        n_cols = int(np.ceil(B/n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)

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

def test_record():
    from rebar import storing
    from . import networks, mcts, analysis, hex

    n_envs = 1
    world = hex.Hex.initial(n_envs=n_envs, boardsize=5, device='cuda')
    network = networks.Network(world.obs_space, world.action_space, width=128).to(world.device)
    network.load_state_dict(storing.load_latest()['network'])
    agent = mcts.MCTSAgent(network, n_nodes=16)

    analysis.record(world, [agent, agent], 20, N=0).notebook()

def test_rollout():
    from . import networks, mcts, mohex
    env = hex.Hex.initial(n_envs=4, boardsize=5, device='cuda')
    network = networks.Network(env.obs_space, env.action_space, width=128).to(env.device)
    agent = mcts.MCTSAgent(env, network, n_nodes=16)
    oppo = mohex.MoHexAgent(env)

    trace = rollout(env, [agent, oppo], 20)

    trace.responses.rewards.sum(0).sum(0)