from .. import cuda

_cache = None
def module():
    global _cache
    if _cache is None:
        _cache = cuda.load(__package__)
    return _cache 

def step(*args, **kwargs):
    return module().step(*args, **kwargs)

def observe(*args, **kwargs):
    return module().observe(*args, **kwargs)