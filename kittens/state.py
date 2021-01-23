import json
from portalocker import RLock
from pathlib import Path
from contextlib import contextmanager
import shutil

ROOT = Path('.kittens')

DEFAULT_STATE = {
    'jobs': {}
}

def path():
    return ROOT / 'state.json'

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
        _lock = RLock(ROOT / '_lock')

    ROOT.mkdir(exist_ok=True, parents=True)
    with _lock:
        yield

def state():
    with lock():
        if not path().exists():
            path().parent.mkdir(exist_ok=True, parents=True)
            path().write_text(json.dumps(DEFAULT_STATE))
        return json.loads(path().read_text())

def jobs(status=None):
    if status:
        return {name: sub for name, sub in jobs().items() if sub['status'] == status}
    return state()['jobs']

@contextmanager
def update():
    with lock():
        s = state()
        yield s
        path().write_text(json.dumps(s))

def mock_dir(f):
    def g(*args, **kwargs):
        global ROOT
        try:
            OLD = ROOT
            ROOT = Path('.kittens-test')
            if ROOT.exists():
                shutil.rmtree(ROOT)
            return f(*args, **kwargs)
        finally:
            ROOT = OLD
    return g
