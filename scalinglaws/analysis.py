import torch
from rebar import arrdict

def apply(agents, inputs):
    indices, actions = [], []
    for i, agent in enumerate(agents):
        m = inputs.seats == i
        if m.any():
            indices.append(m.nonzero().squeeze(1))
            subagent = agent[m]
            actions.append(subagent(inputs[m]).actions)
            agent[m] = subagent
    indices, actions = torch.cat(indices), arrdict.cat(actions)
    return actions[torch.argsort(indices)]

def rollout(env, agents, n_steps):
    inputs = env.reset()
    trace = []
    for _ in range(n_steps):
        actions = apply(agents, inputs)
        responses, new_inputs = env.step(actions)
        trace.append(arrdict.arrdict(
            inputs=inputs,
            actions=actions,
            responses=responses,
            state=env.state_dict()))
        inputs = new_inputs
    return arrdict.stack(trace)

def plot_all(f):

    def proxy(state):
        import numpy as np
        import matplotlib.pyplot as plt

        B = state.seat.shape[0]
        n_rows = int(B**.5)
        n_cols = int(np.ceil(B/n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)

        for e in range(B):
            f(state, e, ax=axes.flatten()[e])
        
        return fig
    return proxy

def record(env, agents, n_steps):
    from rebar.recording import ParallelEncoder
    trace = rollout(env, agents, n_steps)

    state = arrdict.numpyify(trace.state)
    with ParallelEncoder(plot_all(env.plot_state), N=0, fps=1) as encoder:
        for i in range(state.board.shape[0]):
            encoder(state[i])
    return encoder

def test_rollout():
    from . import networks, mcts, mohex
    env = hex.Hex(n_envs=4, boardsize=5, device='cuda')
    network = networks.Network(env.obs_space, env.action_space, width=128).to(env.device)
    agent = mcts.MCTSAgent(env, network, n_nodes=16)
    oppo = mohex.MoHexAgent(env)

    trace = rollout(env, [agent, oppo], 20)

    trace.responses.rewards.sum(0).sum(0)