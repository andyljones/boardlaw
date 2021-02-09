from .. import cuda
from rebar import profiling

_cache = None
def module():
    global _cache
    if _cache is None:
        _cache = cuda.load(__package__)
    return _cache 

@profiling.nvtx
def step(*args, **kwargs):
    return module().step(*args, **kwargs)

@profiling.nvtx
def observe(*args, **kwargs):
    return module().observe(*args, **kwargs)