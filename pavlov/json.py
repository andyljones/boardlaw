import json
from . import runs, files

def write(run, prefix, val):
    filename = f'{prefix}.json'
    if not files.exists(run, filename):
        files.new_file(run, filename)

    pass