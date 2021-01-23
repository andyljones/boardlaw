import os
from . import state, machines
import shutil
import psutil
import tarfile
from subprocess import Popen

#TODO: Is there a way to not create zombies in the first place?
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
    def machines():
        return [{
            'name': 'local',
            'resources': {'gpu': 2, 'memory': 64},
            'processes': [p.info['pid'] for p in psutil.process_iter(['pid', 'status']) if p.info['status'] not in DEAD]}]

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