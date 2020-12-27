import torch
import numpy as np
import inspect
import pandas as pd
from .. import registry
from ... import numpy, tests

KINDS = {}

def clean(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x

class TimeseriesReader:

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
        return resampled.ffill(limit=1).iloc[-1]

class SimpleTimeseriesReader(TimeseriesReader):

    def format(self, rule):
        name = '.'.join(self._key.split('.')[1:])
        final = self.final(rule).item()
        if isinstance(final, int):
            return [(name, f'{final:<6g}')]
        if isinstance(final, float):
            return [(name, f'{final:<6g}')]
        else:
            raise ValueError() 

class ConfidenceTimeseriesReader(TimeseriesReader):

    def format(self, rule):
        name = '.'.join(self._key.split('.')[1:])
        final = self.final(rule)
        return [(name, f'{final.μ:.2f}±{2*final.σ:.2f}')]


def timeseries(reader=SimpleTimeseriesReader):

    def factory(f):
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
    
    return factory