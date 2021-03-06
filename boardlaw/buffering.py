"""
* Going to add a sample at a time
* Need to keep a mask? End index? of whose samples have finished their games
* 
"""
import numpy as np
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

    def __init__(self, length, keep=1.):
        self.length = length
        self._buffer = None
        self.keep = keep

    def update_targets(self, terminal, rewards):
        if terminal.any():
            ts, bs = update_indices(self._ready[terminal], self.current)
            bs = terminal.nonzero(as_tuple=False).squeeze(1)[bs]

            self._buffer.targets[ts % self.length, bs] = rewards[bs]
            self._ready[terminal] = self.current

    def add_raw(self, subset, terminal, rewards):
        if self._buffer is None:
            self.device = terminal.device
            self.n_envs = terminal.size(0)
            self.n_seats = rewards.shape[-1]
            self.ts = torch.arange(self.length, device=self.device)
            self.bs = torch.arange(self.n_envs, device=self.device)

            self._buffer = subset.map(lambda x: x.new_zeros((self.length, *x.shape)))
            self._buffer['targets'] = torch.zeros((self.length, self.n_envs, self.n_seats), device=self.device, dtype=torch.half)

            self.current = 0
            self._ready = torch.zeros((self.n_envs,), device=self.device, dtype=torch.long)

        if np.random.rand() <= self.keep:
            self._buffer[self.current % self.length] = arrdict.arrdict(
                **subset,
                targets=torch.zeros((self.n_envs, self.n_seats), device=self.device, dtype=torch.half))
            self.current = self.current + 1
        self.update_targets(terminal, rewards)

    def add(self, sample):
        """Expects the obs to precede the transition"""
        # Conversions here take a 1600B sample down to a 600B sample 
        subset = arrdict.arrdict(
            obs=sample.worlds.obs.byte(),
            valid=sample.worlds.valid,
            seats=sample.worlds.seats.byte(),
            logits=sample.decisions.logits.half())
        return self.add_raw(subset, sample.transitions.terminal, sample.transitions.rewards.half())

    def ready(self):
        return (self._ready > 0).all()

    def sample(self, size):
        bs = torch.randint(0, self.n_envs, device=self.device, size=(size,))

        rs = torch.rand(device=self.device, size=(size,))
        start = max(self.current - self.length, 0)
        ends = self._ready[bs]
        if (ends == start).any():
            raise ValueError('No ready trajectories to draw from')
        ts = rs*(ends - start) + start

        sample = self._buffer[ts.long() % self.length, bs]
        return arrdict.arrdict(
            obs=sample.obs.float(),
            valid=sample.valid,
            seats=sample.seats.int(),
            logits=sample.logits.float(),
            targets=sample.targets.float())

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
        torch.tensor([[0.]]))
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([True]),
        torch.tensor([[1.]]))
    torch.testing.assert_allclose(
        buffer._buffer.targets,
        torch.tensor([[[1.]], [[1.]], [[0.]]]))

    # Check wrapped behaviour
    buffer = Buffer(3)
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([False]),
        torch.tensor([[0.]]))
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([True]),
        torch.tensor([[0.]]))
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([False]),
        torch.tensor([[0.]]))
    buffer.add_raw(
        arrdict.arrdict(),
        torch.tensor([True]),
        torch.tensor([[1.]]))
    torch.testing.assert_allclose(
        buffer._buffer.targets, 
        torch.tensor([[[1.]], [[0.]], [[1.]]]))

def test_buffer():
    n_envs = 3
    bs = torch.arange(n_envs)
    ts = torch.zeros_like(bs)
    durations = bs+1

    buffer = Buffer(5)

    for _ in range(8):
        terminal = (ts+1) % durations == 0
        rewards = (ts+1)*terminal.float()
        buffer.add_raw(
            arrdict.arrdict(ts=ts, bs=bs),
            terminal,
            rewards[..., None])
        ts = ts + 1

    torch.testing.assert_allclose(
        buffer._buffer.targets,
        torch.tensor([[6., 6., 6.],
                [7., 8., 0.],
                [8., 8., 0.],
                [4., 4., 6.],
                [5., 6., 6.]])[..., None])

    for _ in range(10):
        sample = buffer.sample(3)
        ds = durations[sample.bs]
        expected = (sample.ts // ds + 1)*ds
        torch.testing.assert_allclose(sample.targets, expected[..., None].float())
