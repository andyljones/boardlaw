import time
import numpy as np
from pavlov import storage
import pickle
from . import mcts
import inspect
from logging import getLogger

log = getLogger(__name__)

# Found by inspecting the `main/` runs
BOUNDS = {
    3: (1e10, 5e11),
    4: (1e10, 1e13),
    5: (1e11, 3e13),
    6: (1e11, 4e14),
    7: (1e11, 1e16),
    8: (1e11, 3e16),
    9: (1e12, 1e17)}

TIMES = {
    7: 3600,
}

SAMPLES = {
    3: 1e8,
    4: 2e8,
    5: 3e8,
    6: 6e8,
    7: 1e9,
    8: 1.5e9,
    9: 2e9,
}

def flops_per_sample(agent):
    bound = inspect.signature(mcts.MCTS).bind(world=None, **agent.kwargs)
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

def flops_savepoints(boardsize, n_snapshots=21, upper=None):
    lower = BOUNDS[boardsize][0]
    upper = upper or BOUNDS[boardsize][1]
    return 10**np.linspace(np.log10(lower), np.log10(upper), n_snapshots) 

class FlopsStorer:

    def __init__(self, run, agent):
        self._run = run

        self._flops_per = flops_per_sample(agent)

        boardsize = agent.network.obs_space.dim[0]

        self._savepoints = flops_savepoints(boardsize)
        self._next = 0
        self._n_samples = 0
        self._n_flops = 0

        self._samples_bound = SAMPLES[boardsize]

        storage.save_raw(run, 'model', pickle.dumps(agent.network))

        self._start = time.time()
        self._last_report = time.time()

    def _report(self):
        if time.time() > self._last_report + 60:
            self._last_report = time.time()

            log.info(f'FLOPS: {self._n_flops/self._savepoints[self._next]:.1%} of the way to snapshot #{self._next}') 
            log.info(f'Samples: {self._n_samples/self._samples_bound:.1%} of the way to the end') 

            flops_exp = (time.time() - self._start)/self._n_flops*self._savepoints[-1]
            flops_rem = flops_exp - (time.time() - self._start)

            samples_exp = (time.time() - self._start)/self._n_samples*self._samples_bound
            samples_rem = samples_exp - (time.time() - self._start)

            rem = min(flops_rem, samples_rem)
            secs = rem % 60
            mins = ((rem - secs) // 60) % 60
            hrs = ((rem - secs - 60*mins) // 3600)

            log.info(f'Approx. {hrs:.0f}h{mins:.0f}m{secs:.0f}s remaining')

    def step(self, agent, n_samples):
        self._n_samples += n_samples
        self._n_flops += self._flops_per*n_samples
        sd = {
            'agent': agent.state_dict(), 
            'n_flops': self._n_flops, 
            'n_samples': self._n_samples, 
            'runtime': time.time() - self._start}
        if self._n_flops >= self._savepoints[self._next]:
            log.info(f'Taking a snapshot at {self._n_flops:.1G} FLOPS')
            storage.save_snapshot(self._run, sd)
            self._next += 1

        # For the arena
        #TODO: Swap arena to using snapshots
        storage.throttled_latest(self._run, sd, 60)

        self._report()

        # If there are no more snapshots to take, suggest a break
        flops_overflow = (self._next >= len(self._savepoints))
        sample_overflow = (self._n_samples > self._samples_bound)
        
        return flops_overflow or sample_overflow

def time_savepoints(boardsize, n_snapshots=21):
    return 10**np.linspace(0, np.log10(TIMES[boardsize]), n_snapshots) 

class TimeStorer:

    def __init__(self, run, agent):
        self._run = run

        self._flops_per = flops_per_sample(agent)

        boardsize = agent.network.obs_space.dim[0]

        self._savepoints = time_savepoints(boardsize)
        self._next = 0
        self._n_samples = 0
        self._n_flops = 0

        storage.save_raw(run, 'model', pickle.dumps(agent.network))

        self._start = None

    def step(self, agent, n_samples):
        # Start the timer on the first step, so that warmup time doesn't impact us
        if self._start is None:
            self._start = time.time()
        self._n_samples += n_samples
        self._n_flops += self._flops_per*n_samples
        sd = {
            'agent': agent.state_dict(), 
            'n_flops': self._n_flops, 
            'n_samples': self._n_samples, 
            'runtime': time.time() - self._start}
        if time.time() - self._start >= self._savepoints[self._next]:
            log.info(f'Taking a snapshot')
            storage.save_snapshot(self._run, sd)
            self._next += 1

        # For the arena
        #TODO: Swap arena to using snapshots
        storage.throttled_latest(self._run, sd, 60)

        # If there are no more snapshots to take, suggest a break
        return (self._next >= len(self._savepoints))

