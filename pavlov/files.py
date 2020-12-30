import pandas as pd
import re
import multiprocessing 
import threading
import fnmatch
from . import runs, tests
from pathlib import Path

FILENAME_ORIGIN = r'(?P<origin>.*?)\..*'

def _filename(pattern, extant_files):
    is_pattern = '{n}' in pattern
    count = len([f for _, f in extant_files.items() if f['_pattern'] == pattern])
    if is_pattern:
        return pattern.format(n=count)
    elif count == 0:
        return pattern
    else:
        raise ValueError(f'You\'ve created a "{pattern}" file already, and that isn\'t a valid pattern')

def new_file(run, pattern, **kwargs):
    with runs.update(run) as info:
        filename = _filename(pattern, info['_files'])
        assert re.fullmatch(r'[\w\.-]+', filename), 'Filename contains invalid characters'

        process = multiprocessing.current_process()
        thread = threading.current_thread()
        info['_files'][filename] = {
            '_pattern': pattern,
            '_created': str(tests.timestamp()),
            '_process_id': str(process.pid),
            '_process_name': process.name,
            '_thread_id': str(thread.ident),
            '_thread_name': str(thread.name),
            **kwargs}
    return runs.dir(run) / filename

def info(run, filename):
    return runs.info(run)['_files'][filename]

def path(run, filename):
    return runs.dir(run) / filename

def exists(run, filename):
    return path(run, filename).exists()

def assure(run, filename, default):
    with runs.lock(run):
        isstr = isinstance(default, str)
        p = path(run, filename)
        if not p.exists():
            p.write_text(default) if isstr else p.write_bytes(default)
        return p.read_text() if isstr else p.read_bytes()

def glob(run, glob):
    return {n: i for n, i in runs.info(run)['_files'].items() if fnmatch.fnmatch(n, glob)}

def regex(run, regex):
    return {n: i for n, i in runs.info(run)['_files'].items() if re.fullmatch(regex, n)}

def seq(run, pattern):
    return {n: i for n, i in runs.info(run)['_files'].items() if i['_pattern'] == pattern}

def idx(run, fn):
    pattern = info(run, fn)['_pattern']
    front, back = pattern.split('{n}')
    return int(fn[len(front):-len(back)])

def files(run):
    return runs.info(run)['_files']

def size(run):
    b = sum(path(run, filename).stat().st_size for filename in files(run))
    return b/1e6

def origin(filename):
    if isinstance(filename, Path):
        filename = filename.name
    return re.fullmatch(FILENAME_ORIGIN, filename).group('origin')

def pandas(run=-1):
    df = pd.DataFrame.from_dict(files(run), orient='index')
    df['_created'] = pd.to_datetime(df['_created'])
    df['_origin'] = df.index.map(origin)
    return df.sort_index().sort_index(axis=1)

@tests.mock_dir
def test_new_file():
    run = runs.new_run()
    path = new_file(run, 'test.txt', hello='one')
    name = path.name

    path.write_text('contents')

    i = info(run, name)
    assert i['hello'] == 'one'
    assert path(run, name).read_text()  == 'contents'

@tests.mock_dir
def test_fileglob():
    run = runs.new_run()
    new_file(run, 'foo.txt')
    new_file(run, 'foo.txt')
    new_file(run, 'bar.txt')

    assert len(glob(run, 'foo.txt')) == 2
    assert len(glob(run, 'bar.txt')) == 1