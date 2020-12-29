import pandas as pd
import socket
import threading
import multiprocessing
import re
from contextlib import contextmanager
from pathlib import Path
import json
from portalocker import RLock, AlreadyLocked
import shutil
import pytest
from aljpy import humanhash
from fnmatch import fnmatch
import uuid
from . import tests

ROOT = 'output/pavlov'

### Basic file stuff

def root():
    root = Path(ROOT)
    if not root.exists():
        root.mkdir(exist_ok=True, parents=True)
    return root

def mode(prefix, x):
    if isinstance(x, str):
        return prefix + 't'
    if isinstance(x, bytes):
        return prefix + 'b'
    raise ValueError()

def assert_file(path, default):
    try:
        path.parent.mkdir(exist_ok=True, parents=True)
        with RLock(path, mode('x+', default), fail_when_locked=True) as f:
            f.write(default)
    except (FileExistsError, AlreadyLocked):
        pass

def read(path, mode):
    with RLock(path, mode) as f:
        return f.read()

def read_default(path, default):
    assert_file(path, default)
    return read(path, mode('r', default))

def write(path, contents):
    with RLock(path, mode('w', contents)) as f:
        f.write(contents)

def dir(run, res=True):
    if res:
        run = resolve(run)
    return root() / run

def delete(run):
    assert run != ''
    shutil.rmtree(dir(run))

### Info file stuff

def infopath(run, res=True):
    return dir(run, res) / '_info.json'

def info(run, create=False, res=True):
    path = infopath(run, res)
    if not path.exists():
        raise ValueError(f'Run "{run}" has not been created yet')
    return json.loads(read(path, 'rt'))

def new_info(run, val={}, res=True):
    path = infopath(run, res)
    if path.exists():
        raise ValueError('Info file already exists')
    if not isinstance(val, dict):
        raise ValueError('Info value must be a dict')

    assert_file(path, r'{}')
    write(path, json.dumps(val))
    return path

@contextmanager
def infoupdate(run, create=False):
    # Make sure it's created
    if not infopath(run).exists():
        new_info(run, {})
    # Now grab the lock and do whatever
    with RLock(infopath(run), 'r+t') as f:
        i = json.loads(f.read())
        yield i
        f.truncate(0)
        f.seek(0)
        f.write(json.dumps(i))

### Run stuff

def run_name(suffix='', now=None):
    now = (now or tests.timestamp()).strftime('%Y-%m-%d %H-%M-%S')
    hash = humanhash(str(uuid.uuid4()), n=2)
    return f'{now} {hash} {suffix}'.strip()

def new_run(suffix='', **kwargs):
    now = tests.timestamp()
    run = run_name(suffix, now)
    kwargs = {**kwargs, 
        '_created': str(now), 
        '_host': socket.gethostname(), 
        '_files': {}}
    new_info(run, kwargs, res=False)
    return run

_cache = {}
def runs():
    global _cache

    cache = {}
    for dir in root().iterdir():
        if dir.name in _cache:
            cache[dir.name] = _cache[dir.name]
        else:
            try:
                cache[dir.name] = info(dir.name, res=False) 
            except ValueError:
                # We'll end up here if the run's dir has been created, but 
                # not the info file. That usually happens if we create a 
                # run in another process.
                pass
    
    order = sorted(cache, key=lambda n: cache[n]['_created']) 

    _cache = {n: cache[n] for n in order}
    return _cache

def pandas():
    df = {}
    for run, info in runs().items():
        df[run] = {k: v for k, v in info.items() if k != '_files'}
    df = pd.DataFrame.from_dict(df, orient='index')
    df['_created'] = pd.to_datetime(df['_created'])
    df.index.name = 'run'
    return df

def created(run):
    return pd.to_datetime(info(run)['_created'])

def resolve(run):
    names = list(runs())
    if isinstance(run, int):
        return names[run]
    elif run in names:
        return run
    else: # it's a suffix
        hits = []
        for n in names:
            if n.endswith(run):
                hits.append(n)
        if len(hits) == 1:
            return hits[0]
        else:
            raise ValueError(f'Found {len(hits)} runs that finished with "{run}"')

### Tests

@tests.mock_dir
def test_info():

    # Check reading from a nonexistant file errors
    with pytest.raises(FileNotFoundError):
        info('test')

    # Check trying to write to a nonexistant file errors
    with pytest.raises(FileNotFoundError):
        with infoupdate('test') as (i, writer):
            pass

    # Check we can create a file
    i = new_info('test')
    assert i == {}
    # and read from it
    i = info('test')
    assert i == {}

    # Check we can write to an already-created file
    with infoupdate('test') as (i, writer):
        assert i == {}
        writer({'a': 1})
    # and read it back
    i = info('test')
    assert i == {'a': 1}

    # Check we can write to a not-yet created file
    delete('test')
    with infoupdate('test', create=True) as (i, writer):
        assert i == {}
        writer({'a': 1})
    # and read it back
    i = info('test')
    assert i == {'a': 1}

@tests.mock_dir
def test_new_run():
    run = new_run(desc='test')

    i = info(run)
    assert i['desc'] == 'test'
    assert i['_created']
    assert i['_files'] == {}

@tests.mock_dir
def test_runs():
    fst = new_run('test-1', idx=1)
    snd = new_run('test-2', idx=2)

    i = runs()
    assert len(i) == 2
    assert i[fst]['idx'] == 1
    assert i[snd]['idx'] == 2

