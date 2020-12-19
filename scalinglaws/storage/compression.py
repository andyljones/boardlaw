import shutil
import gzip
import torch
import pickle
import pathlib
from tqdm.auto import tqdm
from rebar import storing
import aljpy

log = aljpy.logger()

def is_optim(d):
    return isinstance(d, dict) and ('state' in d) and ('param_groups' in d)

def strip_optim(d):
    """The optim state is not needed for long-term storage and takes up a lot of space"""
    if isinstance(d, dict):
        return {k: strip_optim(v) for k, v in d.items() if not is_optim(v)}
    return d

def half(d):
    """Can reduce storing by 50% by storing floats as halfs"""
    if isinstance(d, torch.Tensor):
        return d.half()
    elif isinstance(d, dict):
        return {k: half(v) for k, v in d.items()}
    return d

def recompress(path):
    is_compressed = path.open('rb').read(2) == b'\x1f\x8b'
    if is_compressed:
        log.info(f'Don\'t need to compress {path}')
        return
    
    state = pickle.loads(path.read_bytes())
    state = strip_optim(state)
    state = half(state)
    compressed = gzip.compress(pickle.dumps(state))
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(compressed)
    tmp.rename(path)
    log.info(f'Compressed {path}')

def compress_traces():
    groups = storing.stored().groupby('run')
    for r, g in tqdm(groups):
        paths = g.sort_values('period').path
        for path in paths.iloc[:-1]:
            print(f'Unlinking {path}')
            path.unlink()

        log.info(f'Compressing {paths.iloc[-1]}')
        recompress(paths.iloc[-1])
        
def purge(prefix):
    saved = storing.stored()
    for _, row in saved[saved.run.str.startswith(prefix)].iterrows():
        log.info(f'Unlinking {row.path.parent}')
        shutil.rmtree(row.path.parent)
