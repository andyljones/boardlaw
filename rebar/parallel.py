from tqdm.auto import tqdm
from contextlib import contextmanager
import multiprocessing
import types
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, _base, as_completed
import logging
from loky.process_executor import ProcessPoolExecutor as LokyPoolExecutor, _CURRENT_DEPTH, _process_worker, mp
import loky.process_executor

# Disable the extraordinarily frustrating memory leak protection.
loky.process_executor._USE_PSUTIL = False

log = logging.getLogger(__name__)

class SerialExecutor(_base.Executor):
    """An executor that runs things on the main process/thread - meaning stack traces are interpretable
    and the debugger works!
    """
    
    def __init__(self, *args, **kwargs):
        pass
    
    def submit(self, f, *args, **kwargs):
        future = Future()
        future.set_result(f(*args, **kwargs))
        return future

class CUDAPoolExecutor(LokyPoolExecutor):
    # Passes the index of the process to the init, so that we can balance CUDA jobs

    @staticmethod
    def _device_init(n, N):
        import os
        device = n % N
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    def _adjust_process_count(self):
        assert self._initargs == (), 'Device executor doesn\'t currently support custom initializers'
        import torch
        N = torch.cuda.device_count()
        for n in range(len(self._processes), self._max_workers):
            worker_exit_lock = self._context.BoundedSemaphore(1)
            args = (self._call_queue, self._result_queue, self._device_init,
                    (n, N), self._processes_management_lock,
                    self._timeout, worker_exit_lock, _CURRENT_DEPTH + 1)
            worker_exit_lock.acquire()
            try:
                # Try to spawn the process with some environment variable to
                # overwrite but it only works with the loky context for now.
                p = self._context.Process(target=_process_worker, args=args,
                                          env=self._env)
            except TypeError:
                p = self._context.Process(target=_process_worker, args=args)
            p._worker_exit_lock = worker_exit_lock
            p.start()
            self._processes[p.pid] = p
        mp.util.debug('Adjust process count : {}'.format(self._processes))



@contextmanager
def VariableExecutor(N=None, executor='process', **kwargs):
    """An executor that can be easily switched between serial, thread and parallel execution.
    If N=0, a serial executor will be used.
    """
    
    N = multiprocessing.cpu_count() if N is None else N
    
    if N == 0:
        executor = 'serial'

    executors = {
        'loky': LokyPoolExecutor,
        'serial': SerialExecutor,
        'process': ProcessPoolExecutor,
        'thread': ThreadPoolExecutor,
        'cuda': CUDAPoolExecutor}
    executor = executors[executor]
    
    log.debug('Launching a {} with {} processes'.format(executor.__name__, N))    
    with executor(N, **kwargs) as pool:
        yield pool
       
 
@contextmanager
def parallel(f, progress=True, desc=None, **kwargs):
    """Sugar for using the VariableExecutor. Call as
    
    with parallel(f) as g:
        ys = g.wait({x: g(x) for x in xs})
    and f'll be called in parallel on each x, and the results collected in a dictionary.
    A fantastic additonal feature is that if you pass `parallel(f, N=0)` , everything will be run on 
    the host process, so you can `import pdb; pdb.pm()` any errors. 
    """

    with VariableExecutor(**kwargs) as pool:

        def reraise(f, futures={}):
            e = f.exception()
            if e:
                log.warn('Exception raised on "{}"'.format(futures[f]), exc_info=e)
                raise e
            return f.result()

        submitted = set()

        def submit(*args, **kwargs):
            fut = pool.submit(f, *args, **kwargs)
            submitted.add(fut)
            fut.add_done_callback(submitted.discard)  # Try to avoid memory leak
            return fut
        
        def wait(c):
            # Recurse on list-likes
            if type(c) in (list, tuple, types.GeneratorType):
                ctor = list if isinstance(c, types.GeneratorType) else type(c)
                results = wait(dict(enumerate(c)))
                return ctor(results[k] for k in sorted(results))

            # Now can be sure we've got a dict-like
            futures = {fut: k for k, fut in c.items()}
            
            results = {}
            for fut in tqdm(as_completed(futures), total=len(c), disable=not progress, desc=desc):
                results[futures[fut]] = reraise(fut, futures)
                
            return results
        
        def cancel():
            while True:
                remaining = list(submitted)
                for fut in remaining:
                    fut.cancel()
                    submitted.discard(fut)
                if not remaining:
                    break

        try:
            submit.wait = wait
            yield submit
        finally:
            cancel()