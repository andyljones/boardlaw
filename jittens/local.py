import os
from . import jobs, machines
import shutil
import psutil
import tarfile
from subprocess import Popen
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
from shlex import quote

#TODO: Is there a way to just not create zombies in the first place? Double fork?
# I'm actually a bit confused here, because it seems that the *most recent* dead process forms a zombie,
# but older ones don't. Maybe it's something to do with file handles? Total guess that, but half the 
# misery I've had in the past with subprocesses has been related to filehandles.
DEAD = ('zombie',)

@dataclass
class LocalMachine(machines.Machine):
    pass

def add(**kwargs):
    LocalMachine('local', **kwargs, processes=[])
    machines.add('local', type='local', **kwargs)

def allocation_env(allocation):
    env = os.environ.copy()
    for k, vs in allocation.items():
        vals = ",".join(map(str, vs))
        env[f'JITTENS_{k.upper()}'] = vals
    return env

def machine(name, config):
    config = config.copy()
    del config['type']
    assert 'name' not in config
    assert 'processes' not in config
    config['resources'] = {k: list(range(int(v))) for k, v in config['resources'].items()}
    pids = [p.info['pid'] for p in psutil.process_iter(['pid', 'status']) if p.info['status'] not in DEAD]
    return LocalMachine( 
        name=name,
        processes=pids,
        **config)

def launch(job, machine, allocation={}):
    path = Path(machine.root) / job.name
    path.mkdir(parents=True)

    if job.archive:
        tarfile.open(job.archive).extractall(path)

    proc = Popen(
        job.command,
        cwd=path,
        start_new_session=True, 
        shell=True,
        env=allocation_env(allocation))

    return proc.pid

def cleanup(job, machine):
    path = Path(machine.root) / job.name
    if path.exists():
        shutil.rmtree(path)

