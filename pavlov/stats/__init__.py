from . import timeseries
from .. import runs

from .registry import to_run

KINDS = {
    **timeseries.KINDS}

for name, func in KINDS.items():
    globals()[name] = func

def _key(file):
    # file: kind.name.0.ext
    parts = file.split('.')
    return '.'.join(parts[:-2])

def run_readers(run, readers={}):
    readers = {**readers}
    for file, info in runs.files(run).items():
        kind = info['kind']
        if kind in KINDS:
            kind = KINDS[kind]
            k = _key(file)
            if k not in readers:
                readers[k] = kind.Reader(run, k)
    return readers