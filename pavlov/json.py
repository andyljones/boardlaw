import json
from . import runs, files
from portalocker import RLock
from contextlib import contextmanager

def path(run, prefix):
    return files.path(run, f'{prefix}.json')

@contextmanager
def lock(run, prefix):
    lockpath = path(run, prefix).with_suffix('.json.lock')
    with RLock(lockpath):
        yield

def new(run, prefix):
    with lock(run, prefix):
        files.new_file(run, f'{prefix}.json')

@contextmanager
def update(run, prefix):
    with lock(run, prefix):
        contents = path(run, prefix).read_text()
        yield contents
        path.write_text(contents)

def read(run, prefix):
    with lock(run, prefix):
        return json.loads(path(run, prefix).read_text())
