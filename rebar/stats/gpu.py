import torch
import pandas as pd
from io import BytesIO
from subprocess import check_output
from . import writing
import time


def memory(device=0):
    if isinstance(device, torch.device):
        device = device.index
    total_mem = torch.cuda.get_device_properties(f'cuda:{device}').total_memory
    writing.max(f'gpu-memory/{device}/reserve', torch.cuda.max_memory_reserved(device)/total_mem)
    writing.max(f'gpu-memory/{device}/alloc', torch.cuda.max_memory_allocated(device)/total_mem)
    torch.cuda.reset_peak_memory_stats()

def dataframe():
    """Use `nvidia-smi --help-query-gpu` to get a list of query params"""
    params = {
        'device': 'index', 
        'compute': 'utilization.gpu', 'access': 'utilization.memory', 
        'memused': 'memory.used', 'memtotal': 'memory.total',
        'fan': 'fan.speed', 'power': 'power.draw', 'temp': 'temperature.gpu'}
    command = f"""nvidia-smi --format=csv,nounits,noheader --query-gpu={','.join(params.values())}"""
    df = pd.read_csv(BytesIO(check_output(command, shell=True)), header=None)
    df.columns = list(params.keys())
    df = df.set_index('device')
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

_last = -1
def vitals(device=None, throttle=0):
    # This is a fairly expensive op, so let's avoid doing it too often
    global _last
    if time.time() - _last < throttle:
        return
    _last = time.time()

    df = dataframe()
    if device is None:
        pass
    elif isinstance(device, int):
        df = df.loc[[device]]
    elif isinstance(device, torch.device):
        df = df.loc[[device.index]]
    else:
        df = df.loc[device]

    fields = ['compute', 'access', 'fan', 'power', 'temp']
    for (device, field), value in df[fields].stack().iteritems():
        writing.mean(f'gpu/{device}/{field}', value)

    for device in df.index:
        writing.mean(f'gpu-memory/{device}/gross', 100*df.loc[device, 'memused']/df.loc[device, 'memtotal'])