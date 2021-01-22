from datetime import datetime
import json
from portalocker import RLock
from pathlib import Path
from contextlib import contextmanager
from aljpy import humanhash
import shutil

ROOT = 'output/kittens'

DEFAULT_STATE = {
    'submissions': []
}

def path():
    return Path(ROOT) / 'state.json'

_lock = None
@contextmanager
def lock():
    # It's tempting to lock on the state.json file. But that leads to some confusion about 
    # how to handle race conditions when *creating* the _info.json file,
    # and also about how to handle global operations that aren't exclusively
    # about that file. Better to just lock on a purpose-made lock file.

    # Re-entrancy is dealt with on the object rather than on the handle, so need to 
    # keep the object itself about. Ffff.
    global _lock
    if _lock is None:
        _lock = RLock(Path(ROOT) / '_lock')

    Path(ROOT).mkdir(exist_ok=True, parents=True)
    with _lock:
        yield

def state():
    with lock():
        if not path().exists():
            path().parent.mkdir(exist_ok=True, parents=True)
            path().write_text(json.dumps(DEFAULT_STATE))
        return json.loads(path().read_text())

@contextmanager
def update():
    with lock():
        s = state()
        yield s
        path().write_text(json.dumps(s))

def submit(command, archive=None, reqs={}):
    now = datetime.utcnow()
    spec = {
        'name': f'{now.strftime(r"%Y-%m-%d %H-%M-%S")} {humanhash(n=2)}',
        'submitted': str(now),
        'command': command,
        'archive': archive,
        'reqs': reqs,
        'attempts': []}
    with update() as s:
        s['submissions'].append(spec)

def submissions():
    return state()['submissions']

### TESTS

def mock_dir(f):
    def g(*args, **kwargs):
        global ROOT
        try:
            OLD = ROOT
            ROOT = 'output/kittens-test'
            if Path(ROOT).exists():
                shutil.rmtree(ROOT)
            return f(*args, **kwargs)
        finally:
            ROOT = OLD
    return g

@mock_dir
def test():
    submit('test')
    assert len(submissions()) == 1