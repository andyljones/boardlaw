import json
from . import runs, files
from portalocker import RLock
from contextlib import contextmanager

def new(run, prefix):
    pass

@contextmanager
def update(run, prefix):
    path = files.path(run, f'{prefix}.json')
    lockpath = path.with_suffix('.json.lock')
    with RLock(lockpath):
        contents = path.read_text()
        yield contents
        path.write_text(contents)