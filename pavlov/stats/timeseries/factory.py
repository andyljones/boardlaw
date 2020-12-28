import torch
import numpy as np
import inspect
import pandas as pd
from . import plotters, formatters
from .. import registry
from ... import numpy, tests, runs

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

    def __init__(self, run, prefix):
        self._created = runs.created(run)
        self.prefix = prefix
        self._reader = numpy.Reader(run, prefix)
        self._arr = None

    def array(self):
        #TODO: If this gets slow, do amortized allocation of arrays x2 as big as needed
        for name, new in self._reader.read().items():
            parts = [new] if self._arr is None else [self._arr, new]
            joint = np.concatenate(parts)
            self._arr = joint[np.argsort(joint['_time'])]
        return self._arr

    def ready(self):
        return self.array() is not None

    def pandas(self):
        arr = self.array()
        df = pd.DataFrame.from_records(arr, index='_time')
        df.index = df.index.tz_localize('UTC') - self._created
        return df

    def resample(self, rule, **kwargs):
        raw = self.pandas()
        raw = pd.concat([pd.DataFrame(np.nan, [pd.Timedelta(0)], raw.columns), raw])
        raw.index.name = '_time'
        return self.resampler(**raw, rule=rule, **kwargs)

def timeseries(formatter=formatters.simple, plotter=plotters.Simple):

    def factory(f):
        """f provides the signature for the write call, and resamples the saved
        data when it's read."""
        kind = f.__name__

        def write(channel, *args, **kwargs):
            args = tuple(clean(a) for a in args)
            kwargs = {k: clean(v) for k, v in kwargs.items()}

            call = inspect.getcallargs(f, *args, **kwargs)
            del call['kwargs']
            call = {'_time': tests.datetime64(), **call}

            prefix = registry.make_prefix(channel)
            w = registry.writer(prefix, lambda: numpy.Writer(registry.run(), prefix, kind=kind))
            w.write(call)

        reader = type(f'{kind}Reader', (TimeseriesReader,), {
            'resampler': staticmethod(f),
            'format': staticmethod(formatter), 
            'plotter': staticmethod(plotter)})

        write.reader = reader
        KINDS[kind] = write

        return write
    
    return factory