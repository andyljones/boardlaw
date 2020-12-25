import logging
import time
from collections import defaultdict, deque
import logging.handlers
from contextlib import contextmanager
from . import widgets, runs
import sys
import traceback
import _thread
import threading

# for re-export
from logging import getLogger

log = getLogger(__name__)

#TODO: This shouldn't be at the top level
logging.basicConfig(
            stream=sys.stdout, 
            level=logging.INFO, 
            format='%(asctime)s %(levelname)s %(name)s: %(message)s', 
            datefmt=r'%Y-%m-%d %H:%M:%S')
logging.getLogger('parso').setLevel('WARN')  # Jupyter's autocomplete spams the output if this isn't set
log.info('Set log params')

@contextmanager
def handlers(*new_handlers):
    logger = logging.getLogger()
    old_handlers = [*logger.handlers]
    try:
        logger.handlers = new_handlers
        yield 
    finally:
        for h in new_handlers:
            try:
                h.acquire()
                h.flush()
                h.close()
            except (OSError, ValueError):
                pass
            finally:
                h.release()

        logger.handlers = old_handlers

@contextmanager
def to_run(run):
    path = runs.new_file(run, 'logs.txt')
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(name)s: %(message)s', 
        datefmt=r'%H:%M:%S'))

    with handlers(handler):
        try:
            yield
        except:
            log.info(f'Trace:\n{traceback.format_exc()}')
            raise

class Reader:

    def __init__(self, run):
        self._run = run
        self._files = {}

    def read(self):
        for name, info in runs.fileglob(self._run, 'logs.txt').items():
            if name not in self._files:
                path = runs.filepath(self._run, name)
                self._files[name] = (info, path.open('r'))
        
        for name, (info, f) in self._files.items():
            for line in f.readlines():
                yield info, line.rstrip('\n')

def in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

class StdoutRenderer:

    def __init__(self):
        super().__init__()

    def emit(self, info, line):
        source = f'{info["_process_name"]}/#{info["_process_id"]}'
        print(f'{source}: {line}')

    def close(self):
        pass

class IPythonRenderer:

    def __init__(self, compositor=None):
        super().__init__()
        self._out = (compositor or widgets.compositor()).output()
        self._next = time.time()
        self._lasts = {}
        self._buffers = defaultdict(lambda: deque(['']*self._out.lines, maxlen=self._out.lines))

    def _format_block(self, name):
        n_lines = max(self._out.lines//(len(self._buffers) + 2), 1)
        lines = '\n'.join(list(self._buffers[name])[-n_lines:])
        return f'{name}:\n{lines}'

    def _display(self):
        content = '\n\n'.join([self._format_block(n) for n in self._buffers])
        self._out.refresh(content)

        for name, last in list(self._lasts.items()):
            if time.time() - last > 120:
                del self._buffers[name]
                del self._lasts[name]

    def emit(self, info, line):
        source = f'{info["_process_name"]}/#{info["_process_id"]}'
        self._buffers[source].append(line)
        self._lasts[source] = time.time()
        self._display()
    
    def close(self):
        self._display(force=True)
        # Want to leave the outputs open so you can see the final messages
        # self._out.close()
        super().close()

def tail(iterable, n):
    return iter(deque(iterable, maxlen=n))

def from_dir_sync(canceller, run, in_ipython):
    reader = Reader(run)

    if in_ipython:
        renderer = IPythonRenderer()
    else:
        renderer = StdoutRenderer()
    # Get the last ~hundred lines of whatever's there already

    while True:
        for info, line in tail(reader.read(), 100):
            renderer.emit(info, line)

        if canceller.is_set():
            break

        time.sleep(.25)

def _from_dir(*args, **kwargs):
    try:
        from_dir_sync(*args, **kwargs)
    except KeyboardInterrupt:
        log.info('Interrupting main')
        _thread.interrupt_main()
        from_dir_sync(*args, **kwargs)

@contextmanager
def from_run(run):
    try:
        canceller = threading.Event()
        thread = threading.Thread(target=_from_dir, args=(canceller, run, in_ipython))
        thread.start()
        yield
    finally:
        log.info('Cancelling log forwarding thread')
        time.sleep(.25)
        canceller.set()
        thread.join(1)
        if thread.is_alive():
            log.error('Logging thread won\'t die')
        else:
            log.info('Log forwarding thread cancelled')

@contextmanager
def via_run(run, compositor=None):
    with to_run(run), from_run(run, compositor):
        yield

### TESTS

@runs.in_test_dir
def test_in_process():
    run = runs.new_run()
    with from_run(run):
        for _ in range(10):
            log.info('hello')
            time.sleep(.1)

def _test_multiprocess(run):
    with to_run(run):
        for i in range(10):
            log.info(str(i))
            time.sleep(.5)

@runs.in_test_dir
def test_multiprocess():

    import multiprocessing as mp
    run = runs.new_run()
    with from_run(run):
        ps = []
        for _ in range(3):
            p = mp.Process(target=_test_multiprocess, args=(run,))
            p.start()
            ps.append(p)

        while any(p.is_alive() for p in ps):
            time.sleep(.5)

def _test_error(run_name):
    with to_run(run_name):
        log.info('Alive')
        time.sleep(2)
        raise ValueError('Last gasp')

@runs.in_test_dir
def test_error():
    run = runs.new_run()

    import multiprocessing as mp
    with from_run(run):
        ps = []
        for _ in range(1):
            p = mp.Process(target=_test_error, args=(run,))
            p.start()
            ps.append(p)

        while any(p.is_alive() for p in ps):
            time.sleep(.5)
