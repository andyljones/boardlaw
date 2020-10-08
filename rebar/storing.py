from time import strptime
import pandas as pd
import pickle

from pandas._libs.tslibs import Timestamp
from . import paths
import time

def _store(path, objs):
    state_dicts = {k: v.state_dict() for k, v in objs.items()}
    bs = pickle.dumps(state_dicts)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.with_suffix('.tmp').write_bytes(bs)
    path.with_suffix('.tmp').rename(path)

def store_latest(run_name, objs, throttle=0):
    path = paths.process_path(run_name, 'storing', 'latest').with_suffix('.pkl')
    if path.exists():
        if (time.time() - path.lstat().st_mtime) < throttle:
            return False

    _store(path, objs)
    return True

def load_latest(run_name=-1, proc_name='MainProcess'):
    [ps] = list(paths.subdir(run_name, 'storing', 'latest').glob(f'{proc_name}*.pkl'))
    return pickle.loads(ps.read_bytes())

def store_periodic(run_name, objs, throttle=0):
    subdir = paths.subdir(run_name, 'storing', 'periodic')
    subdir.mkdir(exist_ok=True, parents=True)

    now = pd.Timestamp.now().strftime(r'%Y-%m-%d %H-%M-%S')
    path = paths.process_path(run_name, 'storing', 'periodic', now, mkdir=False).with_suffix('.pkl')

    mtime = max([p.lstat().st_mtime for p in subdir.iterdir()], default=0)
    if (time.time() - mtime) < throttle:
        return False

    _store(path, objs)
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

def load_periodic(run_name=-1, proc_name='MainProcess'):
    path = stored_periodic(run_name).loc[lambda df: df.proc_name == proc_name].iloc[-1].path
    return pickle.loads(path.read_bytes())