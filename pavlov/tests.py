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

def time():
    if MOCK_NOW is None:
        return time_.time()
    return MOCK_NOW

def datetime64():
    if MOCK_NOW is None:
        return np.datetime64('now')
    return np.datetime64('1970') + np.timedelta64(MOCK_NOW, 's') 

def timestamp(tz='UTC'):
    if MOCK_NOW is None:
        return pd.Timestamp.now(tz=tz)
    return pd.Timestamp(MOCK_NOW, unit='s', tz=tz)

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
        ROOT = 'output/pavlov-test'
        if Path(ROOT).exists():
            shutil.rmtree(ROOT)

        try:
            result = f(*args, **kwargs)
        finally:
            ROOT = old_ROOT
        return result
    
    return wrapped

### rebar conversion

def convert(run):
    import pandas as pd
    from pathlib import Path
    from collections import defaultdict
    from aljpy import humanhash
    import json

    old = Path(f'output/traces/{run}')
    new = Path(f'output/pavlov/{run} {humanhash(n=2)}')
    new.mkdir(exist_ok=True, parents=True)

    created = pd.to_datetime(run[:19], format='%Y-%m-%d %H-%M-%S').tz_localize('UTC')

    files, counts = {}, defaultdict(lambda: 0)
    for oldpath in old.glob('**/*.npr'):
        parts = oldpath.relative_to(old).parts
        kind = parts[1]
        name = '.'.join(parts[2:-1])
        filename = parts[-1]
        procname = '-'.join(filename.split('-')[:-1])
        procid = filename.split('-')[-1]
        
        newname = f'{kind}.{name}.{counts[kind, name]}.npr'
        
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
        
    created = pd.to_datetime(run[:19], format='%Y-%m-%d %H-%M-%S').tz_localize('UTC')
    info = {
        '_created': str(created), 
        '_files': files}
    (new / '_info.json').write_text(json.dumps(info))

    return new