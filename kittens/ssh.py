from fabric import Connection
from logging import getLogger

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
    return {**config, 'processes': pids}

def launch(job, machine):
    pass

def cleanup(job, machine):
    pass