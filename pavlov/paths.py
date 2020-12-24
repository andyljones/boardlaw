from contextlib import contextmanager
from pathlib import Path
import json
from portalocker import RLock, AlreadyLocked
import shutil
import pytest

ROOT = 'output/pavlov'

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

def infopath(run):
    return dir(run) / 'info.json'

def delete(run):
    assert run != ''
    shutil.rmtree(dir(run))

def info(run, val=None, create=False):
    path = infopath(run)
    if val is None and create:
        return json.loads(read_default(path, r'{}'))
    elif val is None:
        return json.loads(read(path, 'rt'))
    elif create:
        assert isinstance(val, dict)
        assert_file(path, r'{}')
        return write(path, json.dumps(val))
    else:
        assert isinstance(val, dict)
        return write(path, json.dumps(val))

def infos():
    return {dir.name: info(dir.name) for dir in root().iterdir()}

@contextmanager
def infoupdate(run, create=False):
    # Make sure it's created
    info(run, create=create)
    # Now grab the lock and do whatever
    with RLock(infopath(run), 'r+t') as f:
        contents = json.loads(f.read())
        f.truncate(0)
        f.seek(0)
        writer = lambda v: f.write(json.dumps(v))
        yield contents, writer

def use_test_dir(f):

    def wrapped(*args, **kwargs):
        global ROOT
        old_ROOT = ROOT
        ROOT = 'output/pavlov-test'
        if Path(ROOT).exists():
            shutil.rmtree(ROOT)

        try:
            result = f(*args, **kwargs)
        finally:
            ROOT = old_ROOT
        return result
    
    return wrapped

@use_test_dir
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

@use_test_dir
def test_infos():
    info('test-1', {'idx': 1}, create=True)
    info('test-2', {'idx': 2}, create=True)

    i = infos()
    assert len(i) == 2
    assert i['test-1'] == {'idx': 1}
    assert i['test-2'] == {'idx': 2}