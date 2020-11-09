import torch
from rebar import paths, widgets, logging, stats, arrdict
from . import hex, mcts, networks

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

def run():
    env = hex.Hex(n_envs=3, boardsize=3, device='cpu')
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
                
            break
            chunk = as_chunk(buffer)
