from .. import numpy, runs

WRITERS = {}
RUN = None

def to_run(run):
    try:
        global WRITERS, RUN
        old = (WRITERS, RUN)
        WRITERS, RUN = {}, run
        yield
    finally:
        WRITERS, RUN = old

def mean(name, total, count=1):
    key = f'{name}-mean'
    if key not in WRITERS:
        WRITERS[key] = numpy.Writer(RUN, key)
    WRITERS[key].write({'total': total, 'count': count})

