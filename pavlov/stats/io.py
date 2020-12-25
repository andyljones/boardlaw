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

KINDS = {}
def kind(M):
    KINDS[M.__name__.lower()] = M
    return M

@kind
class Mean:

    @staticmethod
    def write(name, total, count=1):
        pattern = f'{name}-mean-{{n}}'
        if pattern not in WRITERS:
            WRITERS[pattern] = numpy.Writer(RUN, pattern, kind='mean')
        WRITERS[pattern].write({'total': total, 'count': count})

    @staticmethod
    def reader(run, name):
        pattern = f'{name}-mean-{{n}}'
        return numpy.MultiReader(run, pattern)

    @staticmethod
    def init(source, read):
        pass


class Reader:

    def __init__(self, run):
        self.run = run

    def read(self):
        for name, info in runs.files(self.run).items():
            path = runs.filepath(self.run, name)
            KINDS[info['kind']].reader(path)
    
    
