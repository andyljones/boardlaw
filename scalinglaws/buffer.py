"""
* Going to add a sample at a time
* Need to keep a mask? End index? of whose samples have finished their games
* 
"""
import torch
from rebar import arrdict

def update_indices(starts, current):
    # This bit of madness is to generate the indices that need to be updated,
    # ie if starts[2] is 7 and current is 10, then the output will have 
    # (7, 8, 9) somewhere in `ts`, and `bs` will have (2, 2, 2) in the 
    # corresponding cells.
    #
    # It'd be a lot easier if PyTorch had a forward-fill.
    counts = current - starts
    ones = starts.new_ones((counts.sum(),))
    cumsums = counts.cumsum(0)[:-1]
    ones[cumsums] = -counts[:-1]+1

    ds = ones.cumsum(0)-1

    bs = torch.zeros_like(ds)
    bs[counts[:-1].cumsum(0)] = starts.new_ones((len(starts)-1,))
    bs = bs.cumsum(0)

    ts = starts[bs] + ds
    return ts, bs

class Buffer:

    def __init__(self, size):
        self._size = size
        self._buffer = None

    def update_targets(self, terminal, rewards):
        starts = self.ready
        if terminal.any():
            ts, bs = update_indices(starts[terminal], self.current)
            bs += terminal.nonzero()[bs]

            self._buffer.targets[ts, bs] = rewards[bs]

    def add(self, sample):
        subset = arrdict.arrdict(
            obs=sample.obs,
            logits=sample.decisions.logits,
            terminal=sample.transitions.terminal,
            rewards=sample.transitions.rewards)
        if self._buffer is None:
            example = arrdict.first_value(sample)
            self.device = example.device
            self.n_envs = example.size(0)

            self._buffer = subset.map(lambda x: x.new_zeros((self._size, *x.shape)))
            self._buffer['targets'] = torch.zeros((self.size, self.n_envs), device=self.device)

            self.current = 0
            self.ready = torch.zeros((self.n_envs,), device=self.device, dtype=torch.long)

        self._buffer[self.current] = subset
        self.update_targets(sample.transitions.terminal, sample.transitions.rewards)
        self.current = self.current + 1 % self._size

    def draw(self, size):
        pass

def test_update_indices():
    starts = torch.tensor([7])
    current = 10
    ts, bs = update_indices(starts, current)
    torch.testing.assert_allclose(ts, torch.tensor([7, 8, 9]))
    torch.testing.assert_allclose(bs, torch.tensor([0, 0, 0]))

    starts = torch.tensor([7, 5])
    current = 10
    ts, bs = update_indices(starts, current)
    torch.testing.assert_allclose(ts, torch.tensor([7, 8, 9, 5, 6, 7, 8, 9]))
    torch.testing.assert_allclose(bs, torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]))

    starts = torch.tensor([9])
    current = 10
    ts, bs = update_indices(starts, current)
    torch.testing.assert_allclose(ts, torch.tensor([9]))
    torch.testing.assert_allclose(bs, torch.tensor([0]))