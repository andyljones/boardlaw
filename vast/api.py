import time
from subprocess import check_output
import pandas as pd
import json
from pathlib import Path
import aljpy
from logging import getLogger

log = getLogger(__name__)

DISK = 10
MAX_DPH = .5
MAX_INSTANCES = 3

OFFER_COLS = [
    'bundled_results',
    'gpu_frac',
    'compute_cap',
    'cpu_cores',
    'cpu_ram',
    'cuda_max_good',
    'dlperf',
    'dph_total',
    'id',
    'num_gpus',
    'machine_id',
    'gpu_name']

STATUS_COLS = [
    'actual_status',
    'status_msg',
    'ssh_host',
    'ssh_port',
    'gpu_frac',
    'compute_cap',
    'cpu_cores',
    'cpu_ram',
    'cuda_max_good',
    'dlperf',
    'dph_total',
    'id',
    'num_gpus',
    'machine_id',
    'gpu_name']

def set_key():
    target = Path('~/.vast_api_key').expanduser()
    if not target.exists():
        key = json.loads(Path('credentials.json').read_text())['vast']
        target.write_text(key)

def run(command):
    set_key()
    while True:
        for _ in range(5):
            s = check_output(f'vast {command}', shell=True).decode()
            if not s.startswith('failed with error 502'):
                return s
        log.info('Hit multiple 502 errors, trying again')

def offers(query=None, cols=OFFER_COLS):
    if cols is not None:
        return offers(query, None)[cols]
    if query is not None:
        return offers(None, cols).query(query)
    js = json.loads(run(f'search offers --raw --storage {DISK}'))
    return pd.DataFrame.from_dict(js).sort_values('dph_total')

def status(label=None, cols=STATUS_COLS):
    if cols is not None:
        return status(label, None)[cols]

    if label:
        s = status()
        if s is None: 
            raise ValueError('No instances')
        elif isinstance(label, int):
            return s.iloc[label]
        else:
            return s.loc[label]

    js = json.loads(run('show instances --raw'))
    if js:
        return pd.DataFrame.from_dict(js).set_index('label')
    else:
        return pd.DataFrame(index=pd.Index([], name='label'), columns=[])

def launch(query=None):
    s = offers(query).iloc[0]
    assert s.dph_total < MAX_DPH
    assert status() is None or len(status()) < MAX_INSTANCES
    label = aljpy.humanhash(n=2)
    resp = run(f'create instance {s.id}'
        ' --image andyljones/boardlaw'
        ' --onstart-cmd "tini -- dev.sh"'
        f' --disk {DISK}'
        f' --label {label}'
        ' --raw') 

    resp = json.loads(resp)
    assert resp['success']
    log.info(f'Launched "{label}"')
    return label

def wait():
    from IPython import display
    while True:
        s = status(cols=None)
        display.clear_output(wait=True)
        if (s['actual_status'] == 'running').all():
            print('Ready')
            break
        else:
            for label, row in s.iterrows():
                duration = time.time() - row['start_date']
                print(f'({duration:.0f}s) {label:15s} {row["actual_status"]}: {row["status_msg"]}')

def destroy(label):
    id = status(label).id
    resp = run(f'destroy instance {id} --raw')
    assert resp.startswith('destroying instance')

