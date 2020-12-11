import torch
from scalinglaws.heads import Tensor
from time import strptime
import pandas as pd
import pickle
from logging import getLogger
from pandas._libs.tslibs import Timestamp
from . import paths
import time
from io import BytesIO

log = getLogger(__name__)

def collapse(state_dicts):
    collapsed = {}
    for prefix, d in state_dicts.items():
        for k, v in d.items():
            collapsed[f'{prefix}.{k}'] = v
    return collapsed

def expand_once(state_dicts):
    d = {}
    for k, v in state_dicts.items():
        parts = k.split('.')
        [head] = parts[:1]
        tail = '.'.join(parts[1:])
        d.setdefault(head, {})[tail] = v
    return d

def _store(path, objs):
    state_dict = collapse({k: v.state_dict() for k, v in objs.items()})
    #TODO: Is there a better way to do this?
    bs = BytesIO()
    torch.save(state_dict, bs)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.with_suffix('.tmp').write_bytes(bs.getvalue())
    path.with_suffix('.tmp').rename(path)

def store_latest(run_name, throttle=0, **objs):
    path = paths.process_path(run_name, 'storing', 'latest').with_suffix('.pkl')
    if path.exists():
        if (time.time() - path.lstat().st_mtime) < throttle:
            return False

    _store(path, objs)
    log.info(f'Stored latest at "{path}"')
    return True

def load_latest(run_name=-1, proc_name='MainProcess', return_modtime=False, device='cpu'):
    [ps] = list(paths.subdir(run_name, 'storing', 'latest').glob(f'{proc_name}*.pkl'))
    modified = pd.Timestamp(ps.lstat().st_mtime, unit='s')
    sd = torch.load(ps, map_location=device)
    return (sd, modified) if return_modtime else sd

def store_periodic(run_name, throttle=0, **objs):
    subdir = paths.subdir(run_name, 'storing', 'periodic')
    subdir.mkdir(exist_ok=True, parents=True)

    now = pd.Timestamp.now().strftime(r'%Y-%m-%d %H-%M-%S')
    path = paths.process_path(run_name, 'storing', 'periodic', now, mkdir=False).with_suffix('.pkl')

    mtime = max([p.lstat().st_mtime for p in subdir.iterdir()], default=0)
    if (time.time() - mtime) < throttle:
        return False

    _store(path, objs)
    log.info(f'Stored periodic at "{path}"')
    return True

def runs():
    return paths.runs()

def stored_periodic(run_name=-1):
    ps = paths.subdir(run_name, 'storing', 'periodic').glob('**/*.pkl')
    infos = []
    for p in ps:
        parsed = paths.parse(p)
        procname, procid = parsed.filename.split('-')
        infos.append({
            **parsed, 
            'parts': '/'.join(parsed.parts),
            'date': pd.to_datetime(parsed.parts[-1], format=r'%Y-%m-%d %H-%M-%S'), 
            'proc_name': procname,
            'proc_id': procid,
            'path': p})

    if len(infos) == 0:
        raise ValueError('No stored data to load')

    df = pd.DataFrame(infos).sort_values('date').reset_index(drop=True)
    return df

def load_periodic(run_name=-1, idx=-1, proc_name='MainProcess'):
    df = stored_periodic(run_name).loc[lambda df: df.proc_name == proc_name]
    if isinstance(idx, int):
        row = df.iloc[idx]
    else:
        raise ValueError('Only implemented integer indexing')
    return pickle.loads(row.path.read_bytes())