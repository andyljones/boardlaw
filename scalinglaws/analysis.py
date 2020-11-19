import time
import numpy as np
import torch
from rebar import arrdict, stats

def apply(world, agents):
    indices, actions = [], []
    for i, agent in enumerate(agents):
        m = world.seats == i
        if m.any():
            indices.append(m.nonzero(as_tuple=False).squeeze(1))
            actions.append(agent(world[m]).actions)
    indices, actions = torch.cat(indices), arrdict.cat(actions)
    return actions[torch.argsort(indices)]

def rollout(world, agents, n_steps=None, n_trajs=None):
    assert n_steps != n_trajs, 'Must specify exactly one of n_steps or n_trajs'

    trace = []
    steps, trajs = 0, 0
    while True:
        actions = apply(world, agents)
        world, trans = world.step(actions)
        trace.append(arrdict.arrdict(
            actions=actions,
            trans=trans,
            world=world))
        steps += 1
        trajs += trans.terminal.sum()
        if (n_steps and (steps >= n_steps)) or (n_trajs and (trajs >= n_trajs)):
            break
    return arrdict.stack(trace)

class Evaluator:

    def __init__(self, world, opponents, n_trajs, throttle=0):
        assert world.n_envs == 1
        assert world.n_seats == len(opponents) + 1
        self.world = arrdict.cat([world for _ in range(n_trajs)])
        self.opponents = opponents

        self.n_trajs = n_trajs

        self.throttle = throttle
        self.last = time.time()

    def rollout(self, agent):
        traces = {}
        for seat in range(self.world.n_seats):
            agents = self.opponents
            agents.insert(seat, agent)
            traces[seat] = rollout(self.world, agents, n_trajs=self.n_trajs) 
        return traces

    def __call__(self, agent):
        if time.time() - self.last < self.throttle:
            return
        self.last = time.time()

        traces = self.rollout(agent)
        for seat, trace in traces.items():
            wins = (trace.trans.rewards[..., seat] == 1).sum()
            trajs = trace.trans.terminal.sum()
            stats.mean(f'eval/{seat}-wins', wins, trajs)

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

def record(world, agents, n_steps, N=0):
    from rebar.recording import ParallelEncoder
    trace = rollout(world, agents, n_steps)

    state = arrdict.numpyify(trace.world)
    with ParallelEncoder(plot_all(world.plot_state), N=N, fps=1) as encoder:
        for i in range(state.board.shape[0]):
            encoder(state[i])
    return encoder

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