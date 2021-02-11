from pavlov import storage, runs
from ..mcts import MCTSAgent
from ..hex import Hex

def agent(run, idx=None, device='cpu'):
    try:
        network = storage.load_raw(run, 'model', device)
        agent = MCTSAgent(network)

        if idx is None:
            sd = storage.load_latest(run)
        else:
            sd = storage.load_snapshot(run, idx)
        agent.load_state_dict(sd['agent'])

        return agent
    except IOError:
        return None

def worlds(run, n_envs, device='cpu'):
    boardsize = runs.info(run)['params']['boardsize']
    return Hex.initial(n_envs, boardsize, device)