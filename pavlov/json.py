import json
from . import runs, files

def update(run, prefix):
    filename = f'{prefix}.json'
    if not files.exists(run, filename):
        files.new_file(run, filename)

    pass