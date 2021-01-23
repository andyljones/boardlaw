import os
from . import state, machines
import shutil
import psutil
import tarfile
from subprocess import Popen

#TODO: Is there a way to just not create zombies in the first place? Double fork?
# I'm actually a bit confused here, because it seems that the *most recent* dead process forms a zombie,
# but older ones don't. Maybe it's something to do with file handles? Total guess that, but half the 
# misery I've had in the past with subprocesses has been related to filehandles.
DEAD = ('zombie',)

def resource_env(j, m):
    env = os.environ.copy()
    for k in j['resources']:
        end = str(m['resources'][k])
        start = m['resources'][k] - j['resources'][k]
        env[f'KITTENS_{k.upper()}'] = f'{start}:{end}'
    return env

def job_path(j):
    return state.ROOT / 'working-dirs' / j['name']

@machines.register
class Local:

    @staticmethod
    def machine(config):
        assert 'name' not in config
        assert 'processes' not in config
        pids = [p.info['pid'] for p in psutil.process_iter(['pid', 'status']) if p.info['status'] not in DEAD]
        return {
            **config,
            'name': 'local',
            'processes': pids}

    @staticmethod
    def launch(j, m):
        path = job_path(j)
        path.mkdir(parents=True)

        if j['archive'] is not None:
            tarfile.open(j['archive']).extractall(path)

        proc = Popen(j['command'], 
            cwd=path,
            start_new_session=True, 
            shell=True,
            env=resource_env(j, m))

        return proc.pid

    @staticmethod
    def cleanup(j, m):
        path = job_path(j)
        if path.exists():
            shutil.rmtree(path)

def mock_config():
    import json
    path = state.ROOT / 'machines' / 'local.json'
    path.parent.mkdir(exist_ok=True, parents=True)

    content = json.dumps({'type': 'local', 'resources': {'gpu': 2, 'memory': 64}})
    path.write_text(content)
