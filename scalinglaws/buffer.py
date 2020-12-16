"""
* Going to add a sample at a time
* Need to keep a mask? End index? of whose samples have finished their games
* 
"""
import torch
import torch.testing
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

    def __init__(self, length):
        self.length = length
        self._buffer = None

    def update_targets(self, terminal, rewards):
        if terminal.any():
            ts, bs = update_indices(self.ready[terminal], self.current)
            bs = terminal.nonzero(as_tuple=False).squeeze(1)[bs]

            self._buffer.targets[ts % self.length, bs] = rewards[bs]

    def add_raw(self, subset, terminal, rewards):
        if self._buffer is None:
            self.device = terminal.device
            self.n_envs = terminal.size(0)
            self.ts = torch.arange(self.length, device=self.device)
            self.bs = torch.arange(self.n_envs, device=self.device)

            self._buffer = subset.map(lambda x: x.new_zeros((self.length, *x.shape)))
            self._buffer['targets'] = torch.zeros((self.length, self.n_envs), device=self.device)

            self.current = 0
            self.ready = torch.zeros((self.n_envs,), device=self.device, dtype=torch.long)


        self._buffer[self.current % self.length] = arrdict.arrdict(
            **subset,
            targets=torch.zeros((self.n_envs,), device=self.device))
        self.current = self.current + 1
        self.update_targets(terminal, rewards)
        self.ready[terminal] = self.current

    def add(self, sample):
        subset = arrdict.arrdict(
            obs=sample.obs,
            logits=sample.decisions.logits,
            terminal=sample.transitions.terminal,
            rewards=sample.transitions.rewards)
        return self.add_raw(subset, sample.transitions.terminal, sample.transitions.rewards)

    def sample(self, size):
        bs = torch.randint(0, self.n_envs, device=self.device, size=(size,))

        rs = torch.rand(device=self.device, size=(size,))
        baseline = self.ready[bs]
        ts = rs*(self.current - baseline) + baseline

        return self._buffer[ts.long() % self.length, bs]

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

def test_add_raw():
    # Check linear behaviour
    buffer = Buffer(3)
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([False]),
        torch.tensor([0.]))
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([True]),
        torch.tensor([1.]))
    torch.testing.assert_allclose(
        buffer._buffer.targets,
        torch.tensor([[1.], [1.], [0.]]))

    # Check wrapped behaviour
    buffer = Buffer(3)
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([False]),
        torch.tensor([0.]))
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([True]),
        torch.tensor([0.]))
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([False]),
        torch.tensor([0.]))
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([True]),
        torch.tensor([1.]))
    torch.testing.assert_allclose(
        buffer._buffer.targets, 
        torch.tensor([[1.], [0.], [1.]]))

def test_buffer():
    n_envs = 3
    bs = torch.arange(n_envs)
    ts = torch.zeros_like(bs)
    durations = bs+1

    buffer = Buffer(10)

    for _ in range(6):
        terminal = (ts+1) % durations == 0
        rewards = (ts+1)*terminal.float()
        buffer.add_raw(
            arrdict.arrdict(ts=ts, bs=bs),
            terminal,
            rewards)
        ts = ts + 1