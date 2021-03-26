import invoke
import os
from . import jobs, machines
import shutil
import psutil
import tarfile
from subprocess import Popen, check_output
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from shlex import quote

#TODO: Is there a way to just not create zombies in the first place? Double fork?
# I'm actually a bit confused here, because it seems that the *most recent* dead process forms a zombie,
# but older ones don't. Maybe it's something to do with file handles? Total guess that, but half the 
# misery I've had in the past with subprocesses has been related to filehandles.
DEAD = ('zombie',)

def worker_env(job, allocation):
    env = os.environ.copy()
    env['JITTENS_PARAMS'] = str(job.params)
    env['JITTENS_NAME'] = job.name

    for k, vs in allocation.items():
        vals = ",".join(map(str, vs))
        env[f'JITTENS_{k.upper()}'] = vals
    return env

@dataclass
class Machine(machines.Machine):
    _processes: Optional[List[int]] = None

    @staticmethod
    def create(**config):
        config = config.copy()
        del config['type']
        assert 'processes' not in config
        config['resources'] = {k: list(range(int(v))) for k, v in config['resources'].items()}
        return Machine(**config)

    @property
    def processes(self):
        if self._processes is None:
            self._processes = [p.info['pid'] for p in psutil.process_iter(['pid', 'status']) if p.info['status'] not in DEAD]
        return self._processes
    
    def launch(self, job, allocation={}):
        path = Path(self.root) / job.name
        path.mkdir(parents=True)

        if job.archive:
            tarfile.open(job.archive).extractall(path)

        proc = Popen(
            job.command,
            cwd=path,
            start_new_session=True, 
            shell=True,
            env=worker_env(job, allocation))

        return proc.pid

    def run(self, command, **kwargs):
        return invoke.context.Context().run(command, **kwargs)

    def cleanup(self, job):
        path = Path(self.root) / job.name
        self.run(f'rm -rf {quote(str(path))} || true')

    def fetch(self, name, source, target):
        source = str(Path(self.root) / name / source)
        command = f"""rsync -r "{source}/" "{target}" """
        return self.run(command, asynchronous=True)

def add(**kwargs):
    # Check that it's actually valid
    Machine.create(name='local', type='local', **kwargs)
    machines.add('local', type='local', **kwargs)
