import torch
from rebar import timer
from .. import networks, mcts, hex

from logging import getLogger

log = getLogger(__name__)

def worldfunc(n_envs, device='cuda'):
    return hex.Hex.initial(n_envs=n_envs, boardsize=7, device=device)

def agentfunc(device='cuda'):
    worlds = worldfunc(n_envs=1, device=device)
    network = networks.Network(worlds.obs_space, worlds.action_space).to(worlds.device)
    # network.trace(worlds)
    return mcts.MCTSAgent(network, n_nodes=64)

def benchmark_experience_collection():
    # Make sure to init cuda before running this 

    torch.manual_seed(0)
    n_envs = 4096
    worlds = worldfunc(n_envs)
    agent = agentfunc()

    with timer.timer(cuda=True) as t:
        for _ in range(16):
            decisions = agent(worlds, value=True)
            new_worlds, transition = worlds.step(decisions.actions)
            worlds = new_worlds
            log.info('actor stepped')
    print(f'{t/(16*n_envs)}/sample')