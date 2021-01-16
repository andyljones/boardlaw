import torch
import numpy as np
import inspect
import pandas as pd
from . import plotters, formatters
from .. import registry
from ... import numpy, tests, runs, files

KINDS = {}

def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(clean(v) for v in x)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    return x

def _collapse(x, prefix=''):
    if isinstance(x, dict):
        for k, v in x.items():
            assert '.' not in k, 'Can\'t have periods in the key'
            assert isinstance(k, str), 'Key must be a string'
            yield from _collapse(v, f'{prefix}{k}.')
    elif isinstance(x, (tuple, list)):
        yield from [(f'{prefix}{i}', v) for i, v in enumerate(x)]
    else:
        yield prefix[:-1], x

def collapse(x):
    return dict(_collapse(x))

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
        if df.columns.str.contains(r'\.').any():
            df.columns = pd.MultiIndex.from_tuples([c.split('.') for c in df.columns])
        df.index = df.index.tz_localize('UTC') - self._created
        return df

    def resample(self, rule, **kwargs):
        raw = self.pandas()
        raw = pd.concat([pd.DataFrame(np.nan, [pd.Timedelta(0)], raw.columns), raw])
        raw.index.name = '_time'
        parts = {k: raw[k] for k in set(raw.columns.get_level_values(0))}
        resampled = self.resampler(**parts, rule=rule, **kwargs)
        return resampled

def call_dict(f, *args, **kwargs):
    call = inspect.getcallargs(f, *clean(args), **clean(kwargs))
    del call['kwargs']
    call = collapse(call)
    return {'_time': tests.datetime64(), **call}

def timeseries(formatter=formatters.simple, plotter=plotters.Simple):

    def factory(f):
        """f provides the signature for the write call, and resamples the saved
        data when it's read."""
        kind = f.__name__

        def write(channel, *args, **kwargs):
            call = call_dict(f, *args, **kwargs)
            prefix = registry.make_prefix(channel)
            if registry.run() is not None:
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

class ArrayReader:

    def __init__(self, run, prefix):
        self.prefix = prefix
        self._run = run
        self._path = files.path(run, prefix + '.npz')

    def array(self):
        #TODO: If this gets slow, do amortized allocation of arrays x2 as big as needed
        return dict(np.load(self._path))

    def ready(self):
        return self._path.exists()

def arrays(formatter=formatters.null, plotter=plotters.Null):

    def factory(f):
        kind = f.__name__

        def write(channel, *args, **kwargs):
            call = call_dict(f, *args, **kwargs)
            filename = registry.make_prefix(channel) + '.npz'
            if registry.run() is not None:
                path = files.path(registry.run(), filename)
                if not path.exists():
                    files.new_file(registry.run(), filename, kind=kind)
                tmp = path.with_suffix('.tmp.npz')
                np.savez(tmp, **call)
                tmp.rename(path)

        reader = type(f'{kind}Reader', (ArrayReader,), {
            'resampler': staticmethod(f),
            'format': staticmethod(formatter), 
            'plotter': staticmethod(plotter)})

        write.reader = reader
        KINDS[kind] = write

        return write
    
    return factory
