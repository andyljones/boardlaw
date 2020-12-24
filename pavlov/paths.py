from pathlib import Path
import json
from portalocker import RLock, AlreadyLocked
import shutil

ROOT = 'output/pavlov'

def root():
    root = Path(ROOT)
    if not root.exists():
        root.mkdir(exist_ok=True, parents=True)
    return root

def assert_file(path, default):
    mode = 'x+t' if isinstance(default, str) else 'x+b'
    try:
        path.parent.mkdir(exist_ok=True, parents=True)
        with RLock(path, mode, fail_when_locked=True) as f:
            f.write(default)
    except (FileExistsError, AlreadyLocked):
        pass

def read_default(path, default):
    assert_file(path, default)
    mode = 'r+t' if isinstance(default, str) else 'r+b'
    with RLock(path, mode) as f:
        return f.read()

def dir(run):
    return root() / run 

def info(run):
    path = dir(run) / 'info.json'
    return json.loads(read_default(path, r'{}'))

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
    i = info('test')
    assert isinstance(i, dict)
    assert len(i) == 0
    assert Path(ROOT).joinpath('test/info.json').exists()