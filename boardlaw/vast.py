# Build and push image to DockerHub (might have to do this manually)
# Find an appropriate vast machine
# Create a machine using their commandline
# Pass onstart script to start the job
# Use vast, ssh, rsync to monitor things (using fabric and patchwork?)
# Manual destroy
import pandas as pd
import json
from subprocess import check_output
from pathlib import Path
import aljpy
from fabric import Connection
from patchwork.transfers import rsync

DISK = 10
MAX_DPH = .5
MAX_INSTANCES = 3

def set_key():
    target = Path('~/.vast_api_key').expanduser()
    if not target.exists():
        key = json.loads(Path('credentials.yml').read_text())['vast']
        target.write_text(key)

def invoke(command):
    set_key()
    return check_output(f'vast {command}', shell=True)

def offers():
    js = json.loads(invoke(f'search offers --raw --storage {DISK}').decode())
    return pd.DataFrame.from_dict(js)

def suggest():
    o = offers()
    viable = o.query('gpu_name == "RTX 2080 Ti" & num_gpus == 1')
    return viable.sort_values('dph_total').iloc[0]

def launch():
    s = suggest()
    assert s.dph_total < MAX_DPH
    assert len(status()) < MAX_INSTANCES
    label = aljpy.humanhash(n=2)
    resp = invoke(f'create instance {s.id}'
        ' --image andyljones/boardlaw'
        f' --disk {DISK}'
        f' --label {label}'
        ' --raw') 
    resp = json.loads(resp)
    assert resp['success']
    return label

def destroy(label):
    id = status(label).id
    resp = invoke(f'destroy instance {id} --raw')
    assert resp.decode().startswith('destroying instance')

def status(label=None):
    if label:
        s = status()
        if len(s): 
            return s.set_index('label').loc[label]
        else:
            raise KeyError(f'No instance with label "{label}"')
    js = json.loads(invoke('show instances --raw').decode())
    return pd.DataFrame.from_dict(js)

def connection(label):
    # Get the vast key into place: `docker cp ~/.ssh/vast_rsa boardlaw:/root/.ssh/`
    # Would be better to use SSH agent forwarding, if vscode's worked reliably :(
    s = status(label)
    return Connection(
        host=s.ssh_host, 
        user='root', 
        port=int(s.ssh_port), 
        connect_kwargs={'key_filename': ['/root/.ssh/vast_rsa']})
    
def ssh_command(label):
    s = status(label)
    print(f'SSH_AUTH_SOCK="" ssh {s.ssh_host} -p {s.ssh_port} -o StrictHostKeyChecking=no -i /root/.ssh/vast_rsa')

def setup(label):
    conn = connection(label)
    conn.run('touch /root/.no_auto_tmux')
    conn.run('rm /etc/banner')
    conn.run('echo PermitUserEnvironment yes >> /etc/ssh/sshd_config')
    conn.run(r"""sed -n "s/PATH='\(.*\)'/PATH=\1:\$PATH/p" ~/.bashrc >> ~/.ssh/environment""")
    
def deploy(label):
    conn = connection(label)
    resp = rsync(conn, 
        source='.',
        target='/code',
        exclude=('.*', 'output', '**/__pycache__', '*.egg-info'),
        strict_host_keys=False)
    conn.run('pip install -e /code/rebar')
    conn.run('pip install -e /code/activelo')