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
        with RLock(path, mode, fail_when_locked=True) as f:
            f.write(default)
    except (FileExistsError, AlreadyLocked):
        pass

def read_default(path, default):
    assert_file(path, default)
    mode = 'r+t' if isinstance(default, str) else 'r+b'
    with RLock(path, mode) as f:
        return f.read()

def index():
    path = root() / 'index.json'
    return json.loads(read_default(path, '[]'))

def directory(run):
    return ROOT 

def filepath(run, tags):
    pass

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
def test_index():
    global ROOT
    ROOT = 'output/pavlov-test'
    idx = index()
    assert isinstance(idx, list)
    assert len(idx) == 0