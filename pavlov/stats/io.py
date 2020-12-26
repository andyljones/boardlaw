from contextlib import contextmanager
from .. import numpy, runs
from functools import wraps

WRITERS = {}
RUN = None

@contextmanager
def to_run(run):
    try:
        global WRITERS, RUN
        old = (WRITERS, RUN)
        WRITERS, RUN = {}, run
        yield
    finally:
        WRITERS, RUN = old


class Monitor:

    def __init__(self, run):
        self.run = run

    def read(self):
        for name, info in runs.files(self.run).items():
            path = runs.filepath(self.run, name)
            KINDS[info['kind']].reader(path)
    
    
