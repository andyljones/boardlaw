"""
CUDA_VISIBLE_DEVICES=1 nsys profile --force-overwrite true -o "output/nsys" -c cudaProfilerApi  -t cublas,nvtx -e EMIT_NVTX=1 python -c "from boardlaw.multinet import *; profile()"

docker cp boardlaw:/code/output/nsys.qdrep ~/Code/tmp/nsys.qdrep

/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli -f -o prof/ncomp python -c "import os; os.environ['EMIT_NVTX'] = '1'; from drones.complete import *; step()"
"""
import os
import aljpy
import torch
from functools import wraps

log = aljpy.logger()

def nvtx(f):
    name = f'{f.__module__}.{f.__qualname__}'
    emit = os.environ.get('EMIT_NVTX') == '1'
    @wraps(f)
    def g(*args, **kwargs):
        if emit:
            torch.cuda.nvtx.range_push(name)
        try:
            return f(*args, **kwargs)
        finally:
            if emit:
                torch.cuda.nvtx.range_pop()
    return g

def nvtxgen(f):
    name = f'{f.__module__}.{f.__qualname__}'
    emit = os.environ.get('EMIT_NVTX') == '1'
    def g(*args, **kwargs):
        if emit:
            torch.cuda.nvtx.range_push(name)
        try:
            return (yield from f(*args, **kwargs))
        finally:
            if emit:
                torch.cuda.nvtx.range_pop()
    return g

def profilable(f):
    @wraps(f)
    def g(*args, **kwargs):
        if os.environ.get('EMIT_NVTX') == '1':
            log.info('Emitting NVTX')
            try:
                torch.cuda.profiler.cudart().cudaProfilerStart()
                with torch.autograd.profiler.emit_nvtx(record_shapes=True):
                    return f(*args, **kwargs)
            finally:
                torch.cuda.profiler.cudart().cudaProfilerStop()
        else:
            return f(*args, **kwargs)
    return g