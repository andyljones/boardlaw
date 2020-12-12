import torch.cuda
import time
from contextlib import contextmanager

def format_time(dt, unit, precision=0):
    # Nicked from timeit
    units = {"ns": 1e-9, "μs": 1e-6, "ms": 1e-3, "s": 1.0}

    if unit is not None:
        scale = units[unit]
    else:
        scales = [(scale, unit) for unit, scale in units.items()]
        scales.sort(reverse=True)
        for scale, unit in scales:
            if dt >= scale:
                break

    return "%.*f%s" % (precision, dt / scale, unit)

class Timer:

    def __init__(self, start=None, end=None, unit=None):
        self._start = start or time.time()
        self._end = end or None
        self._unit = unit

    def stop(self):
        self._end = time.time()
    
    def start(self):
        return self._start

    def end(self):
        return (self._end or time.time())
    
    def time(self):
        return self._end  - self.start()

    def __mul__(self, c):
        duration = (self.end() - self.start())*c
        return Timer(self.start(), self.start() + duration, self._unit)

    def __truediv__(self, c):
        return self*(1/c)
    
    def __repr__(self):
        return format_time(self.time(), self._unit)

    
@contextmanager
def timer(cuda=False, **kwargs):
    if cuda:
        torch.cuda.synchronize()
    timer = Timer(**kwargs)
    try:
        yield timer
    finally:
        if cuda:
            torch.cuda.synchronize()
        timer.stop()