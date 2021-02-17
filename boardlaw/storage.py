import numpy as np
from pavlov import storage
import pickle
from . import mcts
import inspect
from logging import getLogger

log = getLogger(__name__)

# Found by inspecting the `main/` runs
BOUNDS = {
    3: (1e9, 1e12),
    5: (1e10, 1e14),
    7: (1e11, 1e16),
    9: (1e11, 1e17)}

def flops_per_sample(agent):
    bound = inspect.signature(mcts.MCTS).bind(worlds=None, **agent.kwargs)
    bound.apply_defaults()
    n_nodes = bound.arguments['n_nodes']

    count = 0
    for p in agent.network.parameters():
        if p.ndim == 1:
            # We're adding a bias
            count += p.size(0)
        elif p.ndim == 2:
            # We're doing a matmul with a p.size(1)x1 vector
            count += p.size(0)*p.size(1)

    return n_nodes*count

class LogarithmicStorer:

    def __init__(self, run, agent, n_snapshots=21):
        self._run = run

        self._flops_per = flops_per_sample(agent)

        boardsize = agent.network.obs_space.dims[0]
        lower, upper = BOUNDS[boardsize]

        self._savepoints = 10**np.linspace(np.log10(lower), np.log10(upper), n_snapshots) 
        self._next = 0
        self._n_samples = 0
        self._n_flops = 0

        storage.raw(run, 'model', lambda: pickle.dumps(agent.network))

    def step(self, agent, n_samples):
        self._n_samples += n_samples
        self._n_flops += self._flops_per*n_samples
        if self._n_flops >= self._savepoints[self._next]:
            sd = {'agent': agent, 'n_flops': self._n_flops, 'n_samples': self._n_samples}
            log.info(f'Taking a snapshot at {self._n_flops:.1G} FLOPS')
            storage.snapshot(self._run, sd)
            self._next += 1

        # If there are no more snapshots to take, suggest a break
        return (self._next >= len(self._savepoints))
        
