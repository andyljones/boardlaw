import numpy as np
from numpy.lib import format as npformat
from . import runs
from io import BytesIO
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import time

def infer_dtype(exemplar):
    return np.dtype([(k, v.dtype if isinstance(v, np.generic) else type(v)) for k, v in exemplar.items()])

def make_header(dtype):
    """
    Ref: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
    We're doing version 3. Only difference is the zero shape, since we're
    going to deduce the array size from the filesize.
    """
    assert not dtype.hasobject, 'Arrays with objects in get pickled, so can\'t be appended to'

    bs = BytesIO()
    npformat._write_array_header(bs, {
        'descr': dtype.descr, 
        'fortran_order': False, 
        'shape': (0,)},
        version=(3, 0))
    return bs.getvalue()

class FileWriter:

    def __init__(self, path, period=5):
        self._path = Path(path) if isinstance(path, str) else path
        self._file = None
        self._period = period 
        self._next = time.time()
        
    def _init(self, exemplar):
        self._file = self._path.open('wb', buffering=4096)
        self._dtype = infer_dtype(exemplar)
        self._file.write(make_header(self._dtype))
        self._file.flush()

    def write(self, d):
        if self._file is None:
            self._init(d)
        assert set(d) == set(self._dtype.names)
        row = np.array([tuple(v for v in d.values())], self._dtype)
        self._file.write(row.tobytes())
        self._file.flush()

    def close(self):
        self._file.close()
        self._file = None

class Writer:

    def __init__(self, run):
        self._run = run
        self._writers = {}

    def write(self, name, d, **kwargs):
        if name not in self._writers:
            path = runs.new_file(self._run, f'{name}.npr', **kwargs)
            self._writers[name] = FileWriter(path)
        self._writers[name].write(d)

    def close(self):
        for _, w in self._writers.items():
            w.close()
        self._writers = {}

class FileReader:

    def __init__(self, path):
        self._path = Path(path) if isinstance(path, str) else path
        self._file = None

    def _init(self):
        #TODO: Can speed this up with PAG's regex header parser
        self._file = self._path.open('rb')
        version = npformat.read_magic(self._file)
        _, _, dtype = npformat._read_array_header(self._file, version)
        self._dtype = dtype

    def read(self):
        if self._file is None:
            self._init()
        return np.fromfile(self._file, dtype=self._dtype)

    def close(self):
        self._file.close()
        self._file = None

class Reader:

    def __init__(self, run, glob='*.npr'):
        self._run = runs.resolve(run)
        self._glob = glob
        self._readers = {}

    def read(self):
        info = runs.info(self._run)
        for name in runs.fileglob(self._run, self._glob):
            if name not in self._readers:
                pattern = info['_files'][name]['_pattern']
                self._readers[name] = (pattern, FileReader(runs.filepath(self._run, name)))

        results = defaultdict(lambda: {})
        for name, (pattern, reader) in self._readers.items():
            arr = reader.read()
            if len(arr) > 0:
                results[pattern][name] = arr

        return results

@runs.in_test_dir
def test_file_write_read():
    d = {'total': 65536, 'count': 14, '_time': np.datetime64('now')}
    
    run = runs.new_run()
    path = runs.new_file(run, 'test.npr')

    writer = FileWriter(path)
    writer.write(d)

    reader = FileReader(path)
    r = reader.read()

    assert len(r) == 1

@runs.in_test_dir
def test_write_read():
    run = runs.new_run()

    writer = Writer(run)
    writer.write('traj-length', {'total': 65536, 'count': 14, '_time': np.datetime64('now')})
    writer.write('reward', {'total': 50000.5, 'count': 50, '_time': np.datetime64('now')})

    reader = Reader(run)
    r = reader.read()

    assert len(r) == 2
    print(r)