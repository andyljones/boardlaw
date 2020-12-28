import torch
import numpy as np
from . import runs
from io import BytesIO

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

def _store(run, name, objs, **kwargs):
    #TODO: Is there a better way to do this?
    bs = BytesIO()
    torch.save(objs, bs)
    path = runs.filepath(run, name)
    path.with_suffix('.tmp').write_bytes(bs.getvalue())
    path.with_suffix('.tmp').rename(path)

def _load(run, filename):
    path = runs.filepath(run, filename)
    return torch.load(path, map_location='cpu')

def latest(run, val=None):
    name = 'storage.latest.pkl'
    if val is None:
        return _load(run, name)
    else:
        filename = runs.new_file(run, name)
        _store(run, filename, val, kind='storage.latest')
