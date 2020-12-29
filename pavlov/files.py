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
    with runs.update(run) as i:
        filename = _filename(pattern, i['_files'])
        assert re.fullmatch(r'[\w\.-]+', filename), 'Filename contains invalid characters'

        process = multiprocessing.current_process()
        thread = threading.current_thread()
        i['_files'][filename] = {
            '_pattern': pattern,
            '_created': str(tests.timestamp()),
            '_process_id': str(process.pid),
            '_process_name': process.name,
            '_thread_id': str(thread.ident),
            '_thread_name': str(thread.name),
            **kwargs}
    return runs.dir(run) / filename

def fileinfo(run, filename):
    return runs.info(run)['_files'][filename]

def filepath(run, filename):
    return runs.dir(run) / filename

def fileglob(run, glob):
    return {n: i for n, i in runs.info(run)['_files'].items() if fnmatch.fnmatch(n, glob)}

def fileregex(run, regex):
    return {n: i for n, i in runs.info(run)['_files'].items() if re.fullmatch(regex, n)}

def fileseq(run, pattern):
    return {n: i for n, i in runs.info(run)['_files'].items() if i['_pattern'] == pattern}

def fileidx(run, fn):
    pattern = fileinfo(run, fn)['_pattern']
    front, back = pattern.split('{n}')
    return int(fn[len(front):-len(back)])

def files(run):
    return runs.info(run)['_files']

def size(run):
    b = sum(filepath(run, filename).stat().st_size for filename in files(run))
    return b/1e6

def origin(filename):
    if isinstance(filename, Path):
        filename = filename.name
    return re.fullmatch(FILENAME_ORIGIN, filename).group('origin')

@tests.mock_dir
def test_new_file():
    run = runs.new_run()
    path = new_file(run, 'test.txt', hello='one')
    name = path.name

    path.write_text('contents')

    i = fileinfo(run, name)
    assert i['hello'] == 'one'
    assert filepath(run, name).read_text()  == 'contents'

@tests.mock_dir
def test_fileglob():
    run = runs.new_run()
    new_file(run, 'foo.txt')
    new_file(run, 'foo.txt')
    new_file(run, 'bar.txt')

    assert len(fileglob(run, 'foo.txt')) == 2
    assert len(fileglob(run, 'bar.txt')) == 1