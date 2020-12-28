from contextlib import contextmanager
from functools import wraps
import threading

T = threading.local()
T.QUEUE = []
T.DEFER = False

def check(ret):
    if ret is not None:
        raise ValueError('A deferred call returned a value; this should never happen')

@contextmanager
def defer():
    try:
        T.DEFER = True
        yield
    finally:
        for (f, args, kwargs) in T.QUEUE:
            check(f(*args, **kwargs))
        T.QUEUE = []
        T.DEFER = False

def wrap(f):
    
    @wraps(f)
    def deferred(*args, **kwargs):
        if T.DEFER:
            T.QUEUE.append((f, args, kwargs))
        else:
            check(f(*args, **kwargs))
    
    return deferred
