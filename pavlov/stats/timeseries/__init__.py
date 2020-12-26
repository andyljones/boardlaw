import torch
import time
import numpy as np
import inspect
import pandas as pd
from .. import io
from ... import numpy, runs, tests

__all__ = []

def clean(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return 

class Reader:

    def __init__(self, run, key, resampler):
        self._key = key
        self._reader = numpy.Reader(run, key)
        self._arr = None
        self._resampler = resampler

    def array(self):
        #TODO: If this gets slow, do amortized allocation of arrays x2 as big as needed
        for name, arr in self._reader.read().items():
            current = [] if self._arr is None else [self._arr]
            self._arr = np.concatenate(current + arr)
        return self._arr

    def pandas(self):
        arr = self.array()
        df = pd.DataFrame.from_records(arr, index='_time')
        df.index.name = 'time'
        return df

    def final(self, rule):
        df = self.pandas()

        # Base slightly into the future, else by the time the resample actually happens you're 
        # left with an almost-empty last interval.
        base = int(now() % 60) + 5

        resampled = self._resampler(**{k: df[k] for k in df}, rule=rule, base=base)
        final = resampled.ffill(limit=1).iloc[-1]
        return final

class SingleReader(Reader):

    def format(self, rule):
        final = self.final(rule)
        if isinstance(final, int):
            return f'{final}'
        if isinstance(final, float):
            return f'{final:.6g}'
        else:
            raise ValueError() 

def timeseries(f):
    """f provides the signature for the write call, and resamples the saved
    data when it's read."""
    kind = f.__name__

    def write(name, *args, **kwargs):
        args = tuple(clean(a) for a in args)
        kwargs = {k: clean(v) for k, v in kwargs.items()}

        call = inspect.getcallargs(f, *args, **kwargs)
        call = {'_time': tests.datetime64(), **call}

        key = f'{name}.{kind}'
        if key not in io.WRITERS:
            io.WRITERS[key] = numpy.Writer(io.RUN, key, kind=kind)
        io.WRITERS[key].write(call)

    write.Reader = lambda run, name: SingleReader(run, f'{name}.{kind}', f)
    __all__.append(kind)

    return write

@timeseries
def mean(total, count=1, **kwargs):
    return total.resample(**kwargs).mean()/count.resample(**kwargs).mean()

@tests.mock_time
@tests.mock_dir
def test_mean():
    run = runs.new_run()
    with io.to_run(run):
        tests.set_time(0)
        mean('test', 4, 2)
        tests.set_time(1)
        mean('test', 8, 2)

    final = mean.Reader(run, 'test').final('60s')