import torch
from rebar import paths, widgets, logging, stats, arrdict
from . import hex, mcts, networks

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
