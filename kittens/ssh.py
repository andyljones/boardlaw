import re
from fabric import Connection
from logging import getLogger
from . import local
from pathlib import Path
from shlex import quote

getLogger('paramiko').setLevel('WARN')

_connections = {}
def connection(config):
    if config['name'] not in _connections:
        _connections[config['name']] = Connection(**config['connection']) 
    return _connections[config['name']]

def machine(config):
    assert 'processes' not in config
    #TODO: Is there a better way than parsing ps?
    r = connection(config).run('ps -A -o pid=', pty=False, hide='both')
    pids = [int(pid) for pid in r.stdout.splitlines()]
    return {
        'stdout': '/dev/null', 
        'stderr': '/dev/null', 
        'processes': pids,
        **config}

def resource_string(job, machine):
    s = []
    for k in job['resources']:
        assert re.fullmatch(r'[\w\d_]+', k)
        end = machine['resources'][k]
        start = machine['resources'][k] - job['resources'][k]
        s.append(f'KITTENS_{k.upper()}={start}:{end}')
    return ' '.join(s)

def launch(job, machine):
    env = resource_string(job, machine)
    dir = Path(machine['root']) / job['name']
    subcommand = quote(f'mkdir -p {quote(dir)} && cd {quote(dir)} && export {env} && {quote(job["command"])}')
    cmd = f'/bin/bash -c {subcommand} >{quote(machine["stdout"])} 2>{quote(machine["stderr"])} & echo $!'
    r = connection(machine).run(cmd, hide='both')
    return int(r.stdout)

def cleanup(job, machine):
    pass