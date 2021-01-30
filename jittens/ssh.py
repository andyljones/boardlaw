import re
from fabric import Connection
from logging import getLogger
from . import machines, jobs
from pathlib import Path
from shlex import quote
from dataclasses import dataclass, asdict, field
from typing import Dict

getLogger('paramiko').setLevel('WARN')

def resource_string(allocation):
    s = []
    for k, vs in allocation.items():
        vals = ",".join(map(str, vs))
        s.append(f'JITTENS_{k.upper()}={vals}')
    return ' '.join(s)

@dataclass
class Machine(machines.Machine):
    connection_kwargs: Dict = field(default_factory=dict)
    _connection = None

    @staticmethod
    def create(name, config):
        config = config.copy()
        del config['type']
        assert 'processes' not in config
        config['resources'] = {k: list(range(int(v))) for k, v in config['resources'].items()}
        #TODO: Is there a better way than parsing ps?
        return Machine( 
            name=name,
            **config)

    @property
    def connection(self):
        if self._connection is None:
            self._connection = Connection(**self.connection_kwargs) 
        return self._connection

    @property
    def processes(self):
        if self._processes is None:
            r = self.connection().run('ps -A -o pid=', pty=False, hide='both')
            self._processes = [int(pid) for pid in r.stdout.splitlines()]
        return self._processes

    def launch(self, job: jobs.Job, allocation={}):
        env = resource_string(allocation)
        dir = str(Path(self.root) / job.name)

        if job.archive:
            remote_path = f'/tmp/{job.name}'
            self._connection.put(job.archive, remote_path)
            unarchive = f'tar -xzf {quote(remote_path)} && rm {quote(remote_path)} && '
        else:
            unarchive = ''

        setup = (
            f'mkdir -p {quote(dir)} && '
            f'cd {quote(dir)} && '
            f'{unarchive}'
            f'export {env} && '
            f'{job.command}')

        wrapper = (
            f'/bin/bash -c {quote(setup)} '
            '>/tmp/jittens-ssh.log '
            '2>/tmp/jittens-ssh.log '
            f'& echo $!')

        r = self.connection.run(wrapper, hide='both')
        return int(r.stdout)

    def run(self, command):
        self.connection.run(command)

    def cleanup(self, job):
        dir = str(Path(self.root) / job.name)
        self.run(f"rm -rf {quote(dir)}")

def add(name, **kwargs):
    Machine(name, **kwargs, processes=[])
    machines.add(name, type='ssh', **kwargs)
