import os
import shutil
import psutil
from subprocess import Popen
from logging import getLogger
import tempfile
from pathlib import Path
from . import state

log = getLogger(__name__)

TYPES = {}

#TODO: Is there a way to not create zombies in the first place?
DEAD = ('zombie',)

def register(cls):
    TYPES[cls.__name__] = cls
    return cls

def resource_env(j, m):
    env = os.environ.copy()
    for k in j['resources']:
        end = str(m['resources'][k])
        start = m['resources'][k] - j['resources'][k]
        env[f'KITTENS_{k.upper()}'] = f'{start}:{end}'
    return env

def job_path(j):
    return state.ROOT / 'working-dirs' / j['name']

@register
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

def machines():
    ms = {}
    for t, cls in TYPES.items():
        for m in cls.machines():
            m['type'] = t
            ms[m['name']] = m
    return ms

def launch(j, m):
    return TYPES[m['type']].launch(j, m)

def cleanup(j):
    ms = machines()
    if j['machine'] not in ms:
        log.info(f'No cleanup for job {j["name"]} as machine "{j["machine"]}" no longer exists')
        return 

    m = ms[j['machine']]
    TYPES[m['type']].cleanup(j, m)
