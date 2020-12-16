"""
* Going to add a sample at a time
* Need to keep a mask? End index? of whose samples have finished their games
* 
"""
import torch
from rebar import arrdict

class Buffer:

    def __init__(self, size):
        self._size = size
        self._buffer = None

    def update_targets(self):
        terminal = self._buffer.terminal[self.current]
        starts = self.ready
        pass

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
        self.update_targets()
        self.current = self.current + 1 % self._size

    def draw(self, size):
        pass