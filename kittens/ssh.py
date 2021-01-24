import re
from fabric import Connection
from logging import getLogger
from . import machines, state
from pathlib import Path
from shlex import quote
from dataclasses import dataclass, asdict
from typing import Dict

class SSHMachine(machines.Machine):
    connection: Dict

getLogger('paramiko').setLevel('WARN')

_connections = {}
def connection(machine):
    if isinstance(machine, SSHMachine):
        machine = asdict(machine)
    name = machine['name']

    if name not in _connections:
        _connections[name] = Connection(**machine['connection']) 
    return _connections[name]

def machine(config):
    config = config.copy()
    del config['type']
    assert 'processes' not in config
    #TODO: Is there a better way than parsing ps?
    r = connection(config).run('ps -A -o pid=', pty=False, hide='both')
    pids = [int(pid) for pid in r.stdout.splitlines()]
    return SSHMachine( 
        processes=pids,
        **config)

def resource_string(job: state.Job, machine: SSHMachine):
    s = []
    for k in job.resources:
        assert re.fullmatch(r'[\w\d_]+', k)
        end = machine.resources[k]
        start = machine.resources[k] - job.resources[k]
        s.append(f'KITTENS_{k.upper()}={start}:{end}')
    return ' '.join(s)

def launch(job: state.Job, machine: SSHMachine):
    env = resource_string(job, machine)
    dir = str(Path(machine.root) / job.name)

    if job.archive:
        remote_path = f'/tmp/{job.name}'
        connection(machine).put(job.archive, remote_path)
        unarchive = f'tar -xzf {quote(remote_path)} && rm {quote(remote_path)}'
    else:
        unarchive = ''

    subcommand = (
        f'mkdir -p {quote(dir)} &&'
        f'cd {quote(dir)} &&'
        f'{unarchive}'
        f'export {env} &&'
        f'{quote(job.command)}')

    command = (
        f'/bin/bash -c {quote(subcommand)}'
        f'>{quote(machine.stdout)}'
        f'2>{quote(machine.stderr)}'
        f'& echo $!')

    r = connection(machine).run(command, hide='both')
    return int(r.stdout)

def cleanup(job, machine):
    dir = Path(machine.root) / job.name
    connection(machine).run(f"rm -rf {quote(dir)}")