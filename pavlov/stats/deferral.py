from contextlib import contextmanager
from functools import wraps

QUEUE = []
DEFER = False

def check(ret):
    if ret is not None:
        raise ValueError('A deferred call returned a value; this should never happen')

@contextmanager
def defer():
    global DEFER
    try:
        DEFER = True
        yield
    finally:
        for (f, args, kwargs) in QUEUE:
            check(f(*args, **kwargs))
        DEFER = False

def wrap(f):
    
    @wraps(f)
    def deferred(*args, **kwargs):
        if DEFER:
            QUEUE.append((f, args, kwargs))
        else:
            check(f(*args, **kwargs))
    
    return deferred
