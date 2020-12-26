from contextlib import contextmanager

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

def run():
    if RUN is None:
        raise ValueError('No run currently set')
    return RUN

def writer(key, factory=None):
    if factory is not None:
        if key not in WRITERS:
            WRITERS[key] = factory()
    return WRITERS[key]
        