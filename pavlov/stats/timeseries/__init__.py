import torch
import time
import numpy as np
import inspect
import pandas as pd
from .. import registry
from ... import numpy, runs, tests

# __all__ = []

KINDS = {}

def clean(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x

class Reader:

    def __init__(self, run, key, resampler):
        self._key = key
        self._reader = numpy.Reader(run, key)
        self._arr = None
        self._resampler = resampler

    def array(self):
        #TODO: If this gets slow, do amortized allocation of arrays x2 as big as needed
        for name, arr in self._reader.read().items():
            parts = [arr] if self._arr is None else [self._arr, arr]
            self._arr = np.concatenate(parts)
        return self._arr

    def ready(self):
        return self.array() is not None

    def pandas(self):
        arr = self.array()
        df = pd.DataFrame.from_records(arr, index='_time')
        df.index.name = 'time'
        return df

    def final(self, rule):
        df = self.pandas()

        # Offset slightly into the future, else by the time the resample actually happens you're 
        # left with an almost-empty last interval.
        offset = f'{(tests.time() % 60) + 5}s'

        resampled = self._resampler(**{k: df[k] for k in df}, rule=rule, offset=offset)
        final = resampled.ffill(limit=1).iloc[-1]
        return final.item()

class SingleReader(Reader):

    def format(self, rule):
        name = '.'.join(self._key.split('.')[1:])
        final = self.final(rule)
        if isinstance(final, int):
            return [(name, f'{final}')]
        if isinstance(final, float):
            return [(name, f'{final:.6g}')]
        else:
            raise ValueError() 

def timeseries(f, reader=SingleReader):
    """f provides the signature for the write call, and resamples the saved
    data when it's read."""
    kind = f.__name__

    def write(name, *args, **kwargs):
        args = tuple(clean(a) for a in args)
        kwargs = {k: clean(v) for k, v in kwargs.items()}

        call = inspect.getcallargs(f, *args, **kwargs)
        del call['kwargs']
        call = {'_time': tests.datetime64(), **call}

        key = f'{kind}.{name}'
        w = registry.writer(key, lambda: numpy.Writer(registry.run(), key, kind=kind))
        w.write(call)

    write.Reader = lambda run, key: reader(run, key, f)
    KINDS[kind] = write

    return write

@timeseries
def last(x, **kwargs):
    return x.resample(**kwargs).last().ffill()

@timeseries
def max(x, **kwargs):
    return x.resample(**kwargs).max()

@timeseries
def mean(total, count=1, **kwargs):
    return total.resample(**kwargs).mean()/count.resample(**kwargs).mean()

@timeseries
def std(x, **kwargs):
    return x.resample(**kwargs).std()

@timeseries
def cumsum(total=1, **kwargs):
    return total.resample(**kwargs).sum().cumsum()

@timeseries
def timeaverage(x, **kwargs):
    # TODO: To do this properly, I need to get individual per-device streams
    y = x.sort_index()
    dt = y.index.to_series().diff().dt.total_seconds()
    return (y*dt).resample(**kwargs).mean()/dt.resample(**kwargs).mean()

@timeseries
def duty(duration, **kwargs):
    sums = duration.resample(**kwargs).sum()
    periods = sums.index.to_series().diff().dt.total_seconds()
    return sums/periods

@timeseries
def maxrate(duration, count=1, **kwargs):
    return count.resample(**kwargs).mean()/duration.resample(**kwargs).mean()

@timeseries
def rate(count=1, **kwargs):
    counts = count.resample(**kwargs).sum()
    dt = pd.to_timedelta(counts.index.freq).total_seconds()
    dt = min(dt, (count.index[-1] - count.index[0]).total_seconds())
    return counts/dt

@timeseries
def period(count=1, **kwargs):
    counts = count.resample(**kwargs).sum()
    dt = pd.to_timedelta(counts.index.freq).total_seconds()
    dt = min(dt, (count.index[-1] - count.index[0]).total_seconds())
    return dt/counts

#TODO:
# * log_cumsum
# * mean_std
# * dist

@tests.mock_time
@tests.mock_dir
def test_mean():
    from .. import to_run

    run = runs.new_run()
    with to_run(run):
        tests.set_time(0)
        mean('test', 4, 2)
        tests.set_time(1)
        mean('test', 8, 2)

    reader = mean.Reader(run, 'test')
    final = reader.final('60s')
    assert final == 3.

    [label, value] = reader.format('60s')
    assert label == 'test'
    assert value == '3'