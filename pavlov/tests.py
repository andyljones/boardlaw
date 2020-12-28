import time as time_
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

### Time mocking stuff
# Would ideally use FreezeGun, but it's not reliable with numpy/pandas

MOCK_NOW = None
def set_time(s):
    global MOCK_NOW
    assert MOCK_NOW is not None, 'Can only be called from inside a function decorated with `mock_time`'
    MOCK_NOW = s

def timestamp():
    if MOCK_NOW is None:
        return pd.Timestamp.now('UTC')
    return pd.Timestamp(MOCK_NOW, unit='s', tz='UTC')

def time():
    return timestamp().value/1e9

def datetime64():
    # Bit of a mess this, but I don't trust np.datetime64('now') to give
    # me the UTC time rather than system time, and I can't find any 
    # documentation on it. 
    return np.datetime64(timestamp().tz_localize(None))

def mock_time(f):

    def mocked(*args, **kwargs):
        global MOCK_NOW
        MOCK_NOW = 0.
        try:
            return f(*args, **kwargs)
        finally:
            MOCK_NOW = None

    return mocked

### Dir mocking

def mock_dir(f):

    def wrapped(*args, **kwargs):
        from . import runs
        old_ROOT = runs.ROOT
        runs.ROOT = 'output/pavlov-test'
        if Path(runs.ROOT).exists():
            shutil.rmtree(runs.ROOT)

        try:
            result = f(*args, **kwargs)
        finally:
            runs.ROOT = old_ROOT
        return result
    
    return wrapped

### rebar conversion

def convert(run):
    import pandas as pd
    from pathlib import Path
    from collections import defaultdict
    from aljpy import humanhash
    import json

    date = run[:19]
    suffix = run[20:]
    old = Path(f'output/traces/{run}')
    new = Path(f'output/pavlov/{date} {humanhash(n=2)} {suffix}')
    new.mkdir(exist_ok=True, parents=True)

    created = pd.to_datetime(date, format='%Y-%m-%d %H-%M-%S').tz_localize('UTC')

    # Convert the stats files
    files, counts = {}, defaultdict(lambda: 0)
    for oldpath in old.glob('**/*.npr'):
        parts = oldpath.relative_to(old).parts
        kind = parts[1]
        name = '.'.join(parts[2:-1])
        filename = parts[-1]
        procname = '-'.join(filename.split('-')[:-1])
        procid = filename.split('-')[-1]
        
        newname = f'stats.{name}.{counts[kind, name]}.npr'
        
        counts[kind, name] += 1
        
        newpath = new / newname
        newpath.write_bytes(oldpath.read_bytes())
        
        files[newname] = {
            '_pattern': f'{kind}.{name}.{{n}}.npr',
            '_created': str(created),
            '_process_id': str(procid),
            '_process_name': procname,
            '_thread_id': '0',
            '_thread_name': 'main',
            'kind': kind}

    # Convert the log files
    for oldpath in old.glob('**/*.txt'):
        filename = oldpath.name
        procname = '-'.join(filename.split('-')[:-1])
        procid = filename.split('-')[-1]

        newname = f'logs.{counts["logs"]}.txt'
        counts['logs'] += 1

        newpath = new / newname
        newpath.write_text(oldpath.read_text())
        files[newname] = {
            '_pattern': 'logs.{n}.txt',
            '_created': str(created),
            '_process_id': str(procid),
            '_process_name': procname,
            '_thread_id': '0',
            '_thread_name': 'main'}

    for oldpath in old.glob('**/*.pkl'):
        filename = oldpath.name
        procname = '-'.join(filename.split('-')[:-1])
        procid = filename.split('-')[-1]

        if oldpath.parent.name == 'latest':
            saved = pd.Timestamp(oldpath.lstat().st_mtime, unit='ms')
            newname = 'storage.latest.pkl'
            pattern = 'storage.latest.pkl'
        else:
            saved = pd.to_datetime(oldpath.parent.name, format='%Y-%m-%d %H-%M-%S')
            newname = f'storage.snapshot.{counts["snapshot"]}.pkl'
            pattern = 'storage.snapshot.{n}.pkl'
            counts["snapshot"] += 1

        newpath = new / newname
        newpath.write_bytes(oldpath.read_bytes())
        files[newname] = {
            '_pattern': pattern,
            '_created': str(saved),
            '_process_id': str(procid),
            '_process_name': procname,
            '_thread_id': '0',
            '_thread_name': 'main'}

    # Convert the model files
        
    created = pd.to_datetime(run[:19], format='%Y-%m-%d %H-%M-%S').tz_localize('UTC')
    info = {
        '_created': str(created), 
        '_files': files}
    (new / '_info.json').write_text(json.dumps(info))

    return new

def convert_all():
    for p in Path('output/traces').iterdir():
        
        n_saved = len(list(p.glob('**/*.pkl')))
        if (n_saved-1)/4 > 1:
            convert(p.name)