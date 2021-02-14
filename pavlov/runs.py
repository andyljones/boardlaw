import os
from multiprocessing import Value, context
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
from logging import getLogger

log = getLogger(__name__)

ROOT = 'output/pavlov'

### Basic file stuff

def root():
    root = Path(ROOT)
    if not root.exists():
        root.mkdir(exist_ok=True, parents=True)
    return root

def path(run, res=True):
    if res:
        run = resolve(run)
    return root() / run

def delete(run):
    assert run != ''
    shutil.rmtree(path(run))

@contextmanager
def lock(run, res=True):
    # It's tempting to lock on the _info.json file, since that's where 
    # all the run state is kept. But that leads to some confusion about 
    # how to handle race conditions when *creating* the _info.json file,
    # and also about how to handle global operations that aren't exclusively
    # about that file.
    # 
    # Better to just lock on a purpose-made lock file.
    p = path(run, res)
    if not p.exists():
        raise ValueError('Can\'t take lock as run doesn\'t exist')
    with RLock(p / '_lock'):
        yield

### Info file stuff

def infopath(run, res=True):
    return path(run, res) / '_info.json'

def info(run, res=True):
    with lock(run, res):
        path = infopath(run, res)
        if not path.exists():
            raise ValueError(f'Run "{run}" info file has not been created yet')
        return json.loads(path.read_text())

def new_info(run, val={}, res=True):
    path = infopath(run, res)
    path.parent.mkdir(exist_ok=True, parents=True)
    with lock(run, res):
        if path.exists():
            raise ValueError('Info file already exists')
        if not isinstance(val, dict):
            raise ValueError('Info value must be a dict')
        path.write_text(json.dumps(val))
        return path

@contextmanager
def update(run):
    global _cache
    with lock(run):
        path = infopath(run)
        i = json.loads(path.read_text())
        yield i
        path.write_text(json.dumps(i))

        # Invalidate the cache
        _cache = {}

### Run stuff

def new_name(suffix='', now=None):
    now = (now or tests.timestamp()).strftime('%Y-%m-%d %H-%M-%S')
    hash = humanhash(str(uuid.uuid4()), n=2)
    return f'{now} {hash} {suffix}'.strip()

def new_run(suffix='', **kwargs):
    now = tests.timestamp()
    run = new_name(suffix, now)
    kwargs = {**kwargs, 
        '_created': str(now), 
        '_host': socket.gethostname(), 
        '_files': {},
        '_env': dict(os.environ)}
    log.info(f'Created run {run}')
    new_info(run, kwargs, res=False)
    return run

_cache = {}
def runs(pattern=None):
    if pattern is not None:
        res = set(resolutions(pattern))
        return {k: v for k, v in runs().items() if k in res}

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

def pandas(pattern=None):
    df = {}
    for run, info in runs(pattern).items():
        df[run] = {k: v for k, v in info.items() if k != '_files'}
    df = pd.DataFrame.from_dict(df, orient='index')
    df['_created'] = pd.to_datetime(df['_created'])
    df.index.name = 'run'
    return df.sort_index(axis=1)

def created(run):
    return pd.to_datetime(info(run)['_created'])

def resolutions(run):
    names = list(runs())
    if isinstance(run, int):
        return [names[run]]
    elif run in names:
        return [run]
    else: # it's a glob
        return [n for n in names if fnmatch(n, run)]

def resolve(run):
    if run is None:
        return None
    hits = resolutions(run)
    if len(hits) == 0:
        raise ValueError(f'Found no runs that match query "{run}"')
    if len(hits) == 1:
        return hits[0]
    else:
        recent = ', '.join(f'"{h}"' for h in hits[-3:])
        raise ValueError(f'Found {len(hits)} runs that match glob "{run}", such as: {recent}')

def resuffix(old, new):
    oldpath = path(old)
    date, time, salt = oldpath.name.split(' ')[:3]
    newpath = oldpath.parent / f'{date} {time} {salt} {new}'
    oldpath.rename(newpath)
    
def describe(run, desc):
    with update(run) as i:
        i['description'] = desc

def exists(run=-1):
    return path(run).exists()

### Tests

@tests.mock_dir
def test_info():

    # Check reading from a nonexistant file errors
    with pytest.raises(FileNotFoundError):
        info('test')

    # Check trying to write to a nonexistant file errors
    with pytest.raises(FileNotFoundError):
        with update('test') as (i, writer):
            pass

    # Check we can create a file
    i = new_info('test')
    assert i == {}
    # and read from it
    i = info('test')
    assert i == {}

    # Check we can write to an already-created file
    with update('test') as (i, writer):
        assert i == {}
        writer({'a': 1})
    # and read it back
    i = info('test')
    assert i == {'a': 1}

    # Check we can write to a not-yet created file
    delete('test')
    with update('test', create=True) as (i, writer):
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

