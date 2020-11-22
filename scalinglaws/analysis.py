import time
import numpy as np
import torch
from rebar import arrdict, stats, recording
from logging import getLogger

log = getLogger(__name__)

def rollout(worlds, agents, n_steps=None, n_trajs=None):
    assert n_steps != n_trajs, 'Must specify exactly one of n_steps or n_trajs'

    trace = []
    steps, trajs = 0, 0
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
        trajs += transitions.terminal.sum()
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
        self.last = 0

    def rollout(self, agent):
        log.info(f'Evaluating on {self.n_trajs} trajectories...')
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
        results = arrdict.arrdict()
        for seat, trace in traces.items():
            wins = (trace.transitions.rewards[..., seat] == 1).sum()
            trajs = trace.transitions.terminal.sum()
            results[f'eval/{seat}-wins'] = wins/trajs

        with stats.defer():
            for k, v in results.items():
                stats.last(k, v)

        return results

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