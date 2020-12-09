import os
from pathlib import Path
import shutil
import pandas as pd
import re
from . import dotdict
import multiprocessing as mp

ROOT = 'output/traces'

def timestamp(run_suffix):
    if not isinstance(run_suffix, str):
        run_suffix = run_suffix.__name__
    return f'{pd.Timestamp.now().strftime("%Y-%m-%d %H-%M-%S")} {run_suffix}'

def validate(*parts):
    for x in parts:
        #TODO: Why can't I have _ in a path again?
        for c in ['_', os.sep]:
            assert c not in x, f'Can\'t have "{c}" in the file path'

def resolve(run_name):
    if isinstance(run_name, str):
        return run_name
    if isinstance(run_name, int):
        times = {p: p.stat().st_ctime for p in Path(ROOT).iterdir()}
        paths = sorted(times, key=times.__getitem__)
        return paths[run_name].parts[-1]
    raise ValueError(f'Can\'t find a run corresponding to {run_name}')

def run_dir(run_name):
    run_name = resolve(run_name)
    return Path(ROOT) / run_name

def subdir(run_name, *parts):
    path = run_dir(run_name)
    for p in parts:
        path = path / p
    return path

def rename(old, new):
    run_dir(resolve(old)).rename(run_dir(resolve(new)))

def clear(run_name, *parts):
    shutil.rmtree(subdir(run_name, *parts), ignore_errors=True)

def file_path(run_name, *parts, mkdir=True):
    path = subdir(run_name, *parts)
    if mkdir:
        path.parent.mkdir(exist_ok=True, parents=True)
    return path

def process_path(run_name, *parts, mkdir=True):
    proc = mp.current_process()
    return file_path(run_name, *parts, f'{proc.name}-{proc.pid}', mkdir=mkdir)

def glob(run_name, *parts, pattern='*'):
    paths = subdir(run_name, *parts).glob(pattern)
    return sorted(paths, key=lambda p: p.stat().st_mtime)

def parse(path):
    parts = path.relative_to(ROOT).parts
    return dotdict.dotdict(
        run_name=parts[0], 
        parts=parts[1:-1], 
        filename=parts[-1])

def parse_process_path(path):
    parsed = parse(path)
    parsed['procname'], parsed['pid'] = re.match(r'^(.*)-(.*)$', parsed.filename.split('.')[0]).groups()
    return parsed

def runs():
    paths = []
    for p in Path(ROOT).iterdir():
        paths.append({
            'path': p, 
            'created': pd.Timestamp(p.stat().st_ctime, unit='s'),
            'run_name': p.parts[-1]})
    return pd.DataFrame(paths).sort_values('created').reset_index(drop=True)

def size(*parts):
    b = sum(item.stat().st_size for item in subdir(*parts).glob('**/*.*'))
    return b/1e6
