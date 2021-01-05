import numpy as np
import pandas as pd
import threading
from contextlib import contextmanager
from . import timeseries
from .. import runs, files
import aljpy
import re

KINDS = {
    **timeseries.KINDS}

T = threading.local()
T.WRITERS = {}
T.RUN = None

# channel: label or group.la.bel
# prefix: stats.channel
# filename: prefix.3.npr
PREFIX = r'(?P<origin>.*?)\.(?P<channel>.*)'
FILENAME = r'(?P<prefix>.*)\.(?P<idx>.*)\.(?P<ext>.*)'

@contextmanager
def to_run(run):
    run = runs.resolve(run)
    try:
        if hasattr(T, 'run'):
            raise ValueError('Run already set')
        T.WRITERS = {}
        T.RUN = run
        yield
    finally:
        del T.WRITERS
        del T.RUN

def run():
    return T.RUN

def writer(prefix, factory=None):
    if factory is not None:
        if prefix not in T.WRITERS:
            T.WRITERS[prefix] = factory()
    return T.WRITERS[prefix]

def make_prefix(channel):
    return f'stats.{channel}'

def parse_channel(channel):
    parts = channel.split('.')
    if len(parts) == 1:
        return aljpy.dotdict(group=parts[0], label='')
    else:
        return aljpy.dotdict(group=parts[0], label='.'.join(parts[1:]))

def parse_prefix(prefix):
    p = re.fullmatch(PREFIX, prefix).groupdict()
    return aljpy.dotdict(**p, **parse_channel(p['channel']))

def parse_filename(filename):
    p = re.fullmatch(FILENAME, filename).groupdict()
    return aljpy.dotdict(**p, **parse_prefix(p['prefix']))

class StatsReaders:

    def __init__(self, run):
        self._run = run
        self._pool = {}

    def refresh(self):
        for filename, info in files.files(self._run).items():
            if files.origin(filename) == 'stats':
                prefix = parse_filename(filename).prefix
                kind = info['kind']
                if (kind in KINDS) and (prefix not in self._pool):
                    reader = KINDS[kind].reader(self._run, prefix)
                    self._pool[prefix] = reader

    def __getitem__(self, prefix):
        return self._pool[prefix]

    def __iter__(self):
        return iter(self._pool)
        
def reader(run, channel):
    #TODO: This won't generalise!
    prefix = make_prefix(channel)
    exemplar = f'{prefix}.0.npr'
    if not files.exists(run, exemplar):
        raise IOError(f'Run "{run}" has no "{channel}" files')
    kind = files.info(run, exemplar)['kind']
    reader = KINDS[kind].reader(run, prefix)
    return reader

def exists(run, channel):
    prefix = make_prefix(channel)
    exemplar = f'{prefix}.0.npr'
    return files.exists(run, exemplar)

def array(run, channel):
    return reader(run, channel).array()

def pandas(run, channel, field=None, rule='60s', **kwargs):
    r = reader(run, channel)
    if not r.ready():
        raise ValueError(f'Reader for "{run}" "{channel}" is not ready')
    df = r.resample(rule, **kwargs)
    if field is not None:
        df = df[field]
    return df

def compare(rs, *args, **kwargs):
    ns = [n for r in rs for n in runs.resolutions(r)]
    return pd.concat({n: pandas(n, *args, **kwargs) for n in ns}, 1)