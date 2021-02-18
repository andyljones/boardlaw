import time
from .. import widgets, logs, runs, tests, files
from . import registry
import pandas as pd
import threading
from contextlib import contextmanager
import _thread
from logging import getLogger

log = getLogger(__name__)

def adaptive_rule(df):
    timespan = (df.index[-1] - df.index[0]).total_seconds()
    if timespan < 600:
        return '15s'
    elif timespan < 7200:
        return '1min'
    else:
        return '10min'

def expand_columns(df, category, field):
    if isinstance(df, pd.Series):
        return pd.concat({(category, field): df}, 1)
    else:
        df = df.copy()
        df.columns = [(category, f'{field}/{c}') for c in df.columns]
        return df
    
def tdformat(td):
    """How is this not in Python, numpy or pandas?"""
    x = td.total_seconds()
    x, _ = divmod(x, 1)
    x, s = divmod(x, 60)
    if x < 1:
        return f'{s:.0f}s'
    h, m = divmod(x, 60)
    if h < 1:
        return f'{m:.0f}m{s:02.0f}s'
    else:
        return f'{h:.0f}h{m:02.0f}m{s:02.0f}s'

def formatted_pairs(readers, rule):
    pairs = []
    for _, reader in readers.items():
        if reader.ready():
            pairs.extend(reader.format(reader, rule))
    return pairs

def _insert(tree, path, val):
    if len(path) == 1:
        tree[path[0]] = val
    else:
        if path[0] not in tree:
            tree[path[0]] = {}
        _insert(tree[path[0]], path[1:], val)

def _traverse(tree, path=[]):
    for i, k in enumerate(sorted(tree)):
        v = tree[k]
        subpath = path + [i == (len(tree)-1)]
        if isinstance(v, dict):
            yield subpath, k, ''
            yield from _traverse(v, subpath)
        else:
            yield subpath, k, v  

def padding(path):
    chars = []
    for p in path[1:-1]:
        chars.append('   ' if p else '│  ')
    if len(path) > 1:
        chars.append('└─ ' if path[-1] else '├─ ')
    return ''.join(chars)

def treeformat(pairs):
    if len(pairs) == 0:
        return 'No stats yet'

    tree = {}
    for k, v in pairs:
        _insert(tree, k.split('.'), v)

    keys, vals = [], []
    for path, k, v in _traverse(tree):
        keys.append(padding(path) + k)
        vals.append(v)

    keylen = max(map(len, keys))
    keys = [k + ' '*max(keylen-len(k), 0) for k in keys]

    return '\n'.join(f'{k}    {v}' for k, v in zip(keys, vals))

def from_run_sync(run, rule, canceller=None, throttle=1):
    run = runs.resolve(run)
    out = widgets.compositor().output('stats')
    start = pd.Timestamp(runs.info(run)['_created'])
    pool = registry.StatsReaders(run)

    nxt = 0
    while True:
        if tests.time() > nxt:
            nxt = nxt + throttle

            try:
                pool.refresh()
                pairs = formatted_pairs(pool._pool, rule)
                content = treeformat(pairs)

                size = files.size(run)
                age = tests.timestamp() - start
                out.refresh(f'{run}: {tdformat(age)} old, {rule} rule, {size:.0f}MB on disk\n\n{content}')
            except FileNotFoundError:
                log.warn('Got a file not found error.')

        if canceller is not None and canceller.is_set():
            break

        time.sleep(1.)

def _from_run(*args, **kwargs):
    try:
        from_run_sync(*args, **kwargs)
    except KeyboardInterrupt:
        log.debug('Interrupting main')
        _thread.interrupt_main()

@contextmanager
def from_run(run, rule='60s'):
    if logs.in_ipython():
        try:
            canceller = threading.Event()
            thread = threading.Thread(target=_from_run, args=(run, rule, canceller))
            thread.start()
            yield
        finally:
            canceller.set()
            thread.join(2)
            if thread.is_alive():
                log.error('Stat display thread won\'t die')
            else:
                log.debug('Stat display thread cancelled')

            # Want to leave the outputs open so you can see the final stats
            # out.close()
    else:
        log.info('No stats emitted in console mode')
        yield

def test_treeformat():
    pairs = []
    assert treeformat(pairs) == 'No stats yet'

    pairs = [('a', 'b')]
    assert treeformat(pairs) == 'a b'

    pairs = [('a.b', 'c')]
    assert treeformat(pairs) == 'a. \n b c'

    pairs = [('a.b', 'c'), ('a.d', 'e')]
    assert treeformat(pairs) == 'a. \n b c\n d e'

    pairs = [('a.b', 'c'), ('d', 'e')]
    assert treeformat(pairs) == 'a. \n b c\nd  e'
    
@tests.mock_dir
def demo_from_dir():
    from . import to_run, mean

    run = runs.new_run()
    with to_run(run):
        mean('test', 2)
        pass
    from_run_sync(run, '60s')