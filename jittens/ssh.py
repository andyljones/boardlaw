import re
from fabric import Connection
from logging import getLogger
from . import machines, jobs
from pathlib import Path
from shlex import quote
from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class SSHMachine(machines.Machine):
    connection: Dict

getLogger('paramiko').setLevel('WARN')

def add(name, **kwargs):
    SSHMachine(name, **kwargs, processes=[])
    machines.add(name, type='ssh', **kwargs)

_connections = {}
def connection(machine):
    if isinstance(machine, SSHMachine):
        machine = asdict(machine)
    name = machine['name']

    if name not in _connections:
        _connections[name] = Connection(**machine['connection']) 
    return _connections[name]

def machine(name, config):
    config = config.copy()
    del config['type']
    assert 'processes' not in config
    config['resources'] = {k: list(range(int(v))) for k, v in config['resources'].items()}
    #TODO: Is there a better way than parsing ps?
    r = connection({'name': name, **config}).run('ps -A -o pid=', pty=False, hide='both')
    pids = [int(pid) for pid in r.stdout.splitlines()]
    return SSHMachine( 
        name=name,
        processes=pids,
        **config)

def resource_string(allocation):
    s = []
    for k, vs in allocation.items():
        vals = ",".join(map(str, vs))
        s.append(f'JITTENS_{k.upper()}={vals}')
    return ' '.join(s)

def launch(job: jobs.Job, machine: SSHMachine, allocation={}):
    env = resource_string(allocation)
    dir = str(Path(machine.root) / job.name)

    if job.archive:
        remote_path = f'/tmp/{job.name}'
        connection(machine).put(job.archive, remote_path)
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

    r = connection(machine).run(wrapper, hide='both')
    return int(r.stdout)

def cleanup(job, machine):
    dir = str(Path(machine.root) / job.name)
    connection(machine).run(f"rm -rf {quote(dir)}")