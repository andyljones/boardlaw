import threading
from contextlib import contextmanager
from . import timeseries
from .. import runs
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
    try:
        old = (T.WRITERS, T.RUN)
        T.WRITERS, T.RUN = {}, run
        yield
    finally:
        T.WRITERS, T.RUN = old

def run():
    if T.RUN is None:
        raise ValueError('No run currently set')
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
        for filename, info in runs.files(self._run).items():
            if runs.origin(filename) == 'stats':
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
    kind = runs.fileinfo(run, f'{prefix}.0.npr')['kind']
    reader = KINDS[kind].reader(run, prefix)
    return reader

def array(run, channel):
    return reader(run, channel).array()

def pandas(run, channel):
    r = reader(run, channel)
    return r.pandas()