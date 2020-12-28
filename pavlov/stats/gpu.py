import torch
import pandas as pd
from io import BytesIO
from subprocess import check_output
import time
import torch.cuda

def memory(device=0):
    from . import max_percent
    if isinstance(device, torch.device):
        device = device.index
    total_mem = torch.cuda.get_device_properties(f'cuda:{device}').total_memory
    max_percent(f'gpu-memory.{device}.reserve', torch.cuda.max_memory_reserved(device)/total_mem)
    max_percent(f'gpu-memory.{device}.alloc', torch.cuda.max_memory_allocated(device)/total_mem)
    torch.cuda.reset_peak_memory_stats()

def dataframe():
    """Use `nvidia-smi --help-query-gpu` to get a list of query params"""
    params = {
        'device': 'index', 
        'compute': 'utilization.gpu', 'access': 'utilization.memory', 
        'memused': 'memory.used', 'memtotal': 'memory.total',
        'fan': 'fan.speed', 'temp': 'temperature.gpu', 
        'power': 'power.draw', 'powerlimit': 'power.limit'}
    command = f"""nvidia-smi --format=csv,nounits,noheader --query-gpu={','.join(params.values())}"""
    df = pd.read_csv(BytesIO(check_output(command, shell=True)), header=None)
    df.columns = list(params.keys())
    df = df.set_index('device')
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

_last = -1
def gpu(device=None, throttle=0):
    from . import max_percent, mean_percent

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

    for device, row in df.iterrows():
        for field, value in row.iteritems():
            if field in ('compute', 'fan', 'access'):
                mean_percent(f'gpu.{device}.{field}', value/100)
            if field == 'power':
                mean_percent(f'gpu.{device}.{field}', value/row['powerlimit'])
            if field == 'temp':
                mean_percent(f'gpu.{device}.{field}', value/80)

    for device in df.index:
        max_percent(f'gpu-memory.{device}.gross', df.loc[device, 'memused']/df.loc[device, 'memtotal'])
        memory(device)