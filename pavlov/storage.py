import pandas as pd
import torch
import numpy as np
from . import runs, tests
from io import BytesIO

LATEST = 'storage.latest.pkl'
SNAPSHOT = 'storage.snapshot.{n}.pkl'

def flatten(state_dict, depth=np.inf):
    if depth == 0:
        return state_dict
    if not isinstance(state_dict, dict):
        return state_dict

    collapsed = {}
    for prefix, d in state_dict.items():
        for k, v in d.items():
            collapsed[f'{prefix}.{k}'] = flatten(v, depth-1)
    return collapsed

def deepen(state_dict, depth=np.inf):
    if depth == 0:
        return state_dict
    if not isinstance(state_dict, dict):
        return state_dict

    d = {}
    for k, v in state_dict.items():
        parts = k.split('.')
        [head] = parts[:1]
        tail = '.'.join(parts[1:])
        d.setdefault(head, {})[tail] = flatten(v, depth-1)
    return d

def state_dicts(**objs):
    dicts = {}
    for k, v in objs.items():
        if isinstance(v, dict):
            dicts[k] = state_dicts(**v)
        elif hasattr(v, 'state_dict'):
            dicts[k] = v.state_dict()
        else:
            dicts[k] = v
    return dicts

def save(path, objs):
    #TODO: Is there a better way to do this?
    bs = BytesIO()
    torch.save(objs, bs)
    path.with_suffix('.tmp').write_bytes(bs.getvalue())
    path.with_suffix('.tmp').rename(path)

def load(path):
    return torch.load(path, map_location='cpu')

def latest(run, objs):
    path = runs.filepath(run, LATEST)
    if objs:
        if not path.exists():
            runs.new_file(run, LATEST, kind='storage.latest')
        save(path, objs)
    else:
        return load(path)

def throttled_latest(run, objs, throttle):
    if runs.filepath(run, LATEST).exists():
        last = pd.to_datetime(runs.fileinfo(run, LATEST)['_created'])
    else:
        last = pd.Timestamp(0, unit='s', tz='UTC')

    if tests.timestamp() > last + pd.Timedelta(throttle, 's'):
        latest(run, objs)

def snapshot(run, objs):
    name = 'storage.snapshot.{n}.pkl'
    path = runs.new_file(run, name, kind='storage.snapshot')
    save(path, objs)

def snapshots(run):
    return {runs.filepath(run, fn): info for fn, info in runs.fileseq(run, SNAPSHOT).items()}

def throttled_snapshot(run, objs, throttle):
    files = snapshots(run)
    if files:
        last = pd.to_datetime(max(f['_created'] for f in files.values()))
    else:
        last = pd.Timestamp(0, unit='s', tz='UTC')

    if tests.timestamp() > last + pd.Timedelta(throttle, 's'):
        snapshot(run, objs)
