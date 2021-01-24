import os
from . import state, machines
import shutil
import psutil
import tarfile
from subprocess import Popen
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List


#TODO: Is there a way to just not create zombies in the first place? Double fork?
# I'm actually a bit confused here, because it seems that the *most recent* dead process forms a zombie,
# but older ones don't. Maybe it's something to do with file handles? Total guess that, but half the 
# misery I've had in the past with subprocesses has been related to filehandles.
DEAD = ('zombie',)

@dataclass
class LocalMachine(machines.Machine):
    pass

def resource_env(job, machine):
    env = os.environ.copy()
    for k in job.resources:
        end = str(machine.resources[k])
        start = machine.resources[k] - job.resources[k]
        env[f'KITTENS_{k.upper()}'] = f'{start}:{end}'
    return env

def machine(config):
    config = config.copy()
    del config['type']
    assert 'name' not in config
    assert 'processes' not in config
    pids = [p.info['pid'] for p in psutil.process_iter(['pid', 'status']) if p.info['status'] not in DEAD]
    return LocalMachine( 
        name='local',
        processes=pids,
        **config)

def launch(job, machine):
    path = Path(machine.root) / job.name
    path.mkdir(parents=True)

    if job.archive is not None:
        tarfile.open(job.archive).extractall(path)

    proc = Popen(job.command, 
        cwd=path,
        start_new_session=True, 
        shell=True,
        env=resource_env(job, machine))

    return proc.pid

def cleanup(job, machine):
    path = Path(machine.root) / job.name
    if path.exists():
        shutil.rmtree(path)

def mock_config():
    import json
    path = state.ROOT / 'machines' / 'local.json'
    path.parent.mkdir(exist_ok=True, parents=True)

    content = json.dumps({
        'type': 'local', 
        'root': str(state.ROOT / 'local'),
        'resources': {'gpu': 2, 'memory': 64}})
    path.write_text(content)
