"""
* Going to add a sample at a time
* Need to keep a mask? End index? of whose samples have finished their games
* 
"""
import torch
import torch.testing
from rebar import arrdict

def unwrap(starts, current, size):
    overflowed = (starts >= current)
    counts = torch.where(overflowed,
        current + (size - starts),
        current - starts)
    starts = current - counts
    return starts

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

    def __init__(self, length):
        self.length = length
        self._buffer = None

    def update_targets(self, terminal, rewards):
        if terminal.any():
            starts = unwrap(self.ready[terminal], self.current, self.length)
            ts, bs = update_indices(starts, self.current)
            ts = ts % self.length
            bs += terminal.nonzero(as_tuple=False).squeeze(1)[bs]

            self._buffer.targets[ts, bs] = rewards[bs]

    def _add(self, subset, terminal, rewards):
        if self._buffer is None:
            self.device = terminal.device
            self.n_envs = terminal.size(0)
            self.ts = torch.arange(self.length, device=self.device)
            self.bs = torch.arange(self.n_envs, device=self.device)

            self._buffer = subset.map(lambda x: x.new_zeros((self.length, *x.shape)))
            self._buffer['targets'] = torch.zeros((self.length, self.n_envs), device=self.device)

            self.current = 0
            self.ready = torch.zeros((self.n_envs,), device=self.device, dtype=torch.long)


        self._buffer[self.current] = arrdict.arrdict(
            **subset,
            targets=torch.zeros((self.n_envs,), device=self.device))
        self.current = (self.current + 1) % self.length
        self.update_targets(terminal, rewards)
        self.ready[terminal] = self.current

    def add(self, sample):
        subset = arrdict.arrdict(
            obs=sample.obs,
            logits=sample.decisions.logits,
            terminal=sample.transitions.terminal,
            rewards=sample.transitions.rewards)
        return self._add(subset, sample.transitions.terminal, sample.transitions.rewards)

    def draw(self, size):
        ts = torch.rand(device=self.device, size=(size,))
        bs = torch.randint(0, self.n_envs, device=self.device, size=(size,))

        return 


def test_unwrap():
    starts = unwrap(torch.tensor([0, 2]), 1, 3)
    torch.testing.assert_allclose(starts, torch.tensor([0, -1]))

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

def test_add():
    # Check linear behaviour
    buffer = Buffer(3)
    buffer._add(
        arrdict.arrdict(),
        torch.tensor([False]),
        torch.tensor([0.]))
    buffer._add(
        arrdict.arrdict(),
        torch.tensor([True]),
        torch.tensor([1.]))
    torch.testing.assert_allclose(
        buffer._buffer.targets,
        torch.tensor([[1.], [1.], [0.]]))

    # Check wrapped behaviour
    buffer = Buffer(3)
    buffer._add(
        arrdict.arrdict(),
        torch.tensor([False]),
        torch.tensor([0.]))
    buffer._add(
        arrdict.arrdict(),
        torch.tensor([True]),
        torch.tensor([0.]))
    buffer._add(
        arrdict.arrdict(),
        torch.tensor([False]),
        torch.tensor([0.]))
    buffer._add(
        arrdict.arrdict(),
        torch.tensor([True]),
        torch.tensor([1.]))
    torch.testing.assert_allclose(
        buffer._buffer.targets, 
        torch.tensor([[1.], [0.], [1.]]))

def test():
    pass