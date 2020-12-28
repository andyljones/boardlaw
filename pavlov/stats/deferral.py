from contextlib import contextmanager
from functools import wraps
import threading

T = threading.local()

def check(ret):
    if ret is not None:
        raise ValueError('A deferred call returned a value; this should never happen')

@contextmanager
def defer():
    try:
        T.QUEUE = []
        yield
    finally:
        for (f, args, kwargs) in T.QUEUE:
            check(f(*args, **kwargs))
        del T.QUEUE

def wrap(f):
    
    @wraps(f)
    def deferred(*args, **kwargs):
        if hasattr(T, 'QUEUE'):
            T.QUEUE.append((f, args, kwargs))
        else:
            check(f(*args, **kwargs))
    
    return deferred
