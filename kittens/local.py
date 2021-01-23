import os
from . import state
import shutil
import psutil
import tarfile
from subprocess import Popen

#TODO: Is there a way to just not create zombies in the first place? Double fork?
# I'm actually a bit confused here, because it seems that the *most recent* dead process forms a zombie,
# but older ones don't. Maybe it's something to do with file handles? Total guess that, but half the 
# misery I've had in the past with subprocesses has been related to filehandles.
DEAD = ('zombie',)

def resource_env(job, machine):
    env = os.environ.copy()
    for k in job['resources']:
        end = str(machine['resources'][k])
        start = machine['resources'][k] - job['resources'][k]
        env[f'KITTENS_{k.upper()}'] = f'{start}:{end}'
    return env

def job_path(job):
    return state.ROOT / 'working-dirs' / job['name']

def machine(config):
    assert 'name' not in config
    assert 'processes' not in config
    pids = [p.info['pid'] for p in psutil.process_iter(['pid', 'status']) if p.info['status'] not in DEAD]
    return {
        **config,
        'name': 'local',
        'processes': pids}

def launch(job, machine):
    path = job_path(job)
    path.mkdir(parents=True)

    if job['archive'] is not None:
        tarfile.open(job['archive']).extractall(path)

    proc = Popen(job['command'], 
        cwd=path,
        start_new_session=True, 
        shell=True,
        env=resource_env(job, machine))

    return proc.pid

def cleanup(job):
    path = job_path(job)
    if path.exists():
        shutil.rmtree(path)

def mock_config():
    import json
    path = state.ROOT / 'machines' / 'local.json'
    path.parent.mkdir(exist_ok=True, parents=True)

    content = json.dumps({
        'type': 'local', 
        'resources': {'gpu': 2, 'memory': 64}})
    path.write_text(content)
