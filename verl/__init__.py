"""
API idea:
 * Context manager sets some global variables
 * Inc. a network to use, an env to use, and a logger instance 
 * Logger should be called once per learner step with stats need to track test progress
 * If there's no progress in some small multiple of the expected runtime, throw an error on the logger
 * Context manager can catch result
"""
from contextlib import contextmanager
from . import tests, modules
from .common import TestEnd

INSTANCE = None

def instance():
    if INSTANCE is None:
        raise ValueError('Tried to get the test components, but there\'s no test running')
    return INSTANCE

def log(*args, **kwargs):
    return instance().log(*args, **kwargs)

@contextmanager
def conduct(test, **kwargs):
    global INSTANCE
    if INSTANCE is not None:
        print(f'Clearing previous "{type(INSTANCE).__name__}" test')

    try:
        mod = getattr(tests, test)
        INSTANCE = mod.Test(**kwargs)
        yield INSTANCE
    except TestEnd:
        return INSTANCE

    raise ValueError('Test should\'ve thrown a value error; this should not have been reached')