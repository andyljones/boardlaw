import invoke
import re
from fabric import Connection
from logging import getLogger
from . import machines, jobs
from pathlib import Path
from shlex import quote
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List

getLogger('paramiko').setLevel('WARN')

def worker_env(job, allocation):
    s = [f'JITTENS_PARAMS={quote(str(job.params))}', f'JITTENS_NAME={quote(job.name)}']
    for k, vs in allocation.items():
        vals = ",".join(map(str, vs))
        s.append(f'JITTENS_{k.upper()}={vals}')
    return ' '.join(s)

@dataclass
class Machine(machines.Machine):
    connection_kwargs: Dict = field(default_factory=dict)
    _connection: Any = None
    _processes: Optional[List[int]] = None

    @staticmethod
    def create(**config):
        config = config.copy()
        del config['type']
        assert 'processes' not in config
        config['resources'] = {k: list(range(int(v))) for k, v in config['resources'].items()}
        #TODO: Is there a better way than parsing ps?
        return Machine(**config)

    @property
    def connection(self):
        if self._connection is None:
            self._connection = Connection(**self.connection_kwargs) 
        return self._connection

    @property
    def processes(self):
        if self._processes is None:
            r = self.connection.run('ps -A -o pid=', pty=False, hide='both')
            self._processes = [int(pid) for pid in r.stdout.splitlines()]
        return self._processes

    def launch(self, job: jobs.Job, allocation={}):
        env = worker_env(job, allocation)
        dir = str(Path(self.root) / job.name)

        if job.archive:
            remote_path = f'/tmp/{job.name}'
            self.connection.put(job.archive, remote_path)
            unarchive = f'tar -xzf {quote(remote_path)} && rm {quote(remote_path)} && '
        else:
            unarchive = ''

        # Is there a better way to do this than string-bashing? Especially the env-passing
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

    def run(self, command, **kwargs):
        return self.connection.run(command, **kwargs)

    def cleanup(self, job):
        dir = str(Path(self.root) / job.name)
        self.run(f"rm -rf {quote(dir)}")

    def fetch(self, name, source, target):
        source = str(Path(self.root) / name / source)

        conn = self.connection
        [keyfile] = conn.connect_kwargs['key_filename']
        ssh = f"ssh -o StrictHostKeyChecking=no -i '{keyfile}' -p {conn.port}"

        # https://unix.stackexchange.com/questions/104618/how-to-rsync-over-ssh-when-directory-names-have-spaces
        command = f"""rsync -r -e "{ssh}" {conn.user}@{conn.host}:"'{source}/'" "{target}" """
        return invoke.context.Context().run(command, asynchronous=True)

def add(name, **kwargs):
    Machine(name=name, **kwargs)
    machines.add(name, type='ssh', **kwargs)
