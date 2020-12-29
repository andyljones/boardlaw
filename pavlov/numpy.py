import numpy as np
from numpy.lib import format as npformat
from . import runs, tests, files
from io import BytesIO
from pathlib import Path

FILEPATTERN = '{prefix}.{{n}}.npr'

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

class Writer:

    def __init__(self, run, prefix, **kwargs):
        self._path = files.new_file(run, FILEPATTERN.format(prefix=prefix), **kwargs)
        self._file = None
        self._next = tests.time()
        
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

class MonoReader:

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

class Reader:

    def __init__(self, run, prefix):
        self._run = runs.resolve(run)
        self._pattern = FILEPATTERN.format(prefix=prefix)
        self._readers = {}

    def read(self):
        for name, info in files.seq(self._run, self._pattern).items():
            if name not in self._readers:
                self._readers[name] = MonoReader(files.path(self._run, name))

        results = {}
        for name, reader in self._readers.items():
            arr = reader.read()
            if len(arr) > 0:
                results[name] = arr

        return results

@tests.mock_dir
def test_write_read():
    d = {'total': 65536, 'count': 14, '_time': tests.datetime64()}
    
    run = runs.new_run()

    writer = Writer(run, 'test')
    writer.write(d)

    reader = Reader(run, 'test')
    r = reader.read()

    assert len(r) == 1