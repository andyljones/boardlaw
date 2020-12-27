from contextlib import contextmanager
from . import timeseries
from .. import runs

KINDS = {
    **timeseries.KINDS}

WRITERS = {}
RUN = None

@contextmanager
def to_run(run):
    try:
        global WRITERS, RUN
        old = (WRITERS, RUN)
        WRITERS, RUN = {}, run
        yield
    finally:
        WRITERS, RUN = old

def run():
    if RUN is None:
        raise ValueError('No run currently set')
    return RUN

def writer(key, factory=None):
    if factory is not None:
        if key not in WRITERS:
            WRITERS[key] = factory()
    return WRITERS[key]
        
def _key(file):
    # file: kind.name.0.ext
    parts = file.split('.')
    return '.'.join(parts[:-2])

class ReaderPool:

    def __init__(self, run):
        self.run = run
        self.pool = {}

    def refresh(self):
        for file, info in runs.files(self.run).items():
            kind = info['kind']
            if kind in KINDS:
                kind = KINDS[kind]
                k = _key(file)
                if k not in self.pool:
                    self.pool[k] = kind.reader(self.run, k)