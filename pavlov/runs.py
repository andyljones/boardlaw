import numpy as np
import time as time_
import threading
import multiprocessing
import re
import pandas as pd
from contextlib import contextmanager
from pathlib import Path
import json
from portalocker import RLock, AlreadyLocked
import shutil
import pytest
from aljpy import humanhash
from fnmatch import fnmatch
import string
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

def dir(run):
    return root() / run

def delete(run):
    assert run != ''
    shutil.rmtree(dir(run))

### Info file stuff

def infopath(run):
    return dir(run) / '_info.json'

def info(run, val=None, create=False):
    path = infopath(run)
    if not create and not path.exists():
        raise ValueError(f'Run "{run}" has not been created yet')
    if val is not None and not isinstance(val, dict):
        raise ValueError('Info value must be None or a dict')

    if val is None and create:
        return json.loads(read_default(path, r'{}'))
    elif val is None:
        return json.loads(read(path, 'rt'))
    elif create:
        assert_file(path, r'{}')
        write(path, json.dumps(val))
        return path
    else:
        write(path, json.dumps(val))
        return path

@contextmanager
def infoupdate(run, create=False):
    # Make sure it's created
    info(run, create=create)
    # Now grab the lock and do whatever
    with RLock(infopath(run), 'r+t') as f:
        i = json.loads(f.read())
        yield i
        f.truncate(0)
        f.seek(0)
        f.write(json.dumps(i))

### Run creation stuff

def run_name(suffix='', now=None):
    now = (now or tests.timestamp()).strftime('%Y-%m-%d %H-%M-%S')
    hash = humanhash(str(uuid.uuid4()), n=2)
    return f'{now} {hash} {suffix}'

def resolve(run):
    #TODO: Implement indexing
    return run

def new_run(suffix=None, **kwargs):
    now = tests.timestamp()
    run = run_name(suffix, now)
    kwargs = {**kwargs, '_created': str(now), '_files': {}}
    info(run, kwargs, create=True)
    return run

def runs():
    return {dir.name: info(dir.name) for dir in root().iterdir()}

### File stuff

def _filename(pattern, extant_files):
    is_pattern = any(name == "n" for _, name, _, _ in string.Formatter().parse(pattern))
    count = len([f for _, f in extant_files.items() if f['_pattern'] == pattern])
    if is_pattern:
        return pattern.format(n=count)
    elif count == 0:
        return pattern
    else:
        raise ValueError(f'You\'ve created a "{pattern}" file already, and that isn\'t a valid pattern')

def new_file(run, pattern, **kwargs):
    with infoupdate(run) as i:
        name = _filename(pattern, i['_files'])

        process = multiprocessing.current_process()
        thread = threading.current_thread()
        i['_files'][name] = {
            '_pattern': pattern,
            '_created': str(tests.timestamp()),
            '_process_id': str(process.pid),
            '_process_name': process.name,
            '_thread_id': str(thread.ident),
            '_thread_name': str(thread.name),
            **kwargs}
    return dir(run) / name

def fileinfo(run, name):
    return info(run)['_files'][name]

def filepath(run, name):
    return dir(run) / name

def fileglob(run, pattern):
    return {n: i for n, i in info(run)['_files'].items() if fnmatch(n, pattern)}

def files(run):
    return info(run)['_files']

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
    i = info('test', create=True)
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

@tests.mock_dir
def test_new_file():
    run = new_run()
    path = new_file(run, 'test.txt', hello='one')
    name = path.name

    path.write_text('contents')

    i = fileinfo(run, name)
    assert i['hello'] == 'one'
    assert filepath(run, name).read_text()  == 'contents'

@tests.mock_dir
def test_fileglob():
    run = new_run()
    new_file(run, 'foo.txt')
    new_file(run, 'foo.txt')
    new_file(run, 'bar.txt')

    assert len(fileglob(run, 'foo.txt')) == 2
    assert len(fileglob(run, 'bar.txt')) == 1