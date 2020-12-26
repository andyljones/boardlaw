import time
from .. import widgets, logging, runs, tests
from . import run_readers
import pandas as pd
import threading
from contextlib import contextmanager
import _thread

log = logging.getLogger(__name__)

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
    
def sourceinfo(df):
    names = df.columns.get_level_values(1)
    tags = names.str.extract(r'^(?P<chart1>.*?)/(?P<label>.*)|(?P<chart2>.*)$')
    tags['category'] = df.columns.get_level_values(0)
    tags['key'] = df.columns.get_level_values(1)
    tags['title'] = tags.chart1.combine_first(tags.chart2)
    tags['id'] = tags['category'] + '_' + names
    tags.index = df.columns
    info = tags[['category', 'key', 'title', 'label', 'id']].fillna('')
    return info

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
            pairs.extend(reader.format(rule))
    return pairs

def _insert(tree, path, val):
    if len(path) == 1:
        tree[path[0]] = val
    else:
        if path[0] not in tree:
            tree[path[0]] = {}
        _insert(tree[path[0]], path[1:], val)

def _traverse(tree, depth=0):
    for k in sorted(tree):
        v = tree[k]
        if isinstance(v, dict):
            yield depth, f'{k}.', ''
            yield from _traverse(v, depth+1)
        else:
            yield depth, k, v  

def treeformat(pairs):
    if len(pairs) == 0:
        return 'No stats yet'

    tree = {}
    for k, v in pairs:
        _insert(tree, k.split('.'), v)

    keys, vals = [], []
    for depth, k, v in _traverse(tree):
        keys.append(' '*depth + k)
        vals.append(v)

    keylen = max(map(len, keys))
    keys = [k + ' '*max(keylen-len(k), 0) for k in keys]

    return '\n'.join(f'{k} {v}' for k, v in zip(keys, vals))

def from_dir_sync(run, rule, canceller=None, throttle=1):
    run = runs.resolve(run)
    out = widgets.compositor().output()
    start = pd.Timestamp(runs.info(run)['_created'])
    readers = {}

    nxt = 0
    while True:
        if tests.time() > nxt:
            nxt = nxt + throttle

            readers = run_readers(run, readers)
            pairs = formatted_pairs(readers, rule)
            content = treeformat(pairs)

            size = runs.size(run)
            age = tests.timestamp() - start
            out.refresh(f'{run}: {tdformat(age)} old, {rule} rule, {size:.0f}MB on disk\n\n{content}')

        if canceller is not None and canceller.is_set():
            break

        time.sleep(1.)

def _from_dir(*args, **kwargs):
    try:
        from_dir_sync(*args, **kwargs)
    except KeyboardInterrupt:
        log.info('Interrupting main')
        _thread.interrupt_main()

@contextmanager
def from_dir(run, rule='60s'):
    if logging.in_ipython():
        try:
            canceller = threading.Event()
            thread = threading.Thread(target=_from_dir, args=(run, rule, canceller))
            thread.start()
            yield
        finally:
            canceller.set()
            thread.join(1)
            if thread.is_alive():
                log.error('Stat display thread won\'t die')
            else:
                log.info('Stat display thread cancelled')

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
    from_dir_sync(run, '60s')