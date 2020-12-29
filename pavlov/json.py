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
        yield json.loads(contents)
        path.write_text(json.dumps(contents))

def assure(run, prefix, default):
    #TODO: This hangs because it seems to be... not actually recurrent?
    with lock(run, prefix):
        p = path(run, prefix)
        if not p.exists():
            new(run, prefix)
            p.write_text(json.dumps(default))

def read(run, prefix, default=None):
    with lock(run, prefix):
        p = path(run, prefix)
        if p.exists():
            return json.loads(path(run, prefix).read_text())
        if default is not None:
            return default
        raise FileNotFoundError(f'No file for "{run}" "{prefix}" and no default provided')
