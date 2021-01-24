from logging import getLogger
import jittens
from .api import launch, status, offers, wait, destroy

log = getLogger(__name__)

def jittenate():
    jittens.clear()
    for name, row in status().iterrows():
        if row.actual_status == 'running':
            jittens.ssh.add(name,
                resources={
                    'gpu': row.num_gpus,
                    'memory': row.cpu_ram*row.gpu_frac},
                root='/code',
                connection={
                    'host': row.ssh_host, 
                    'user': 'root', 
                    'port': int(row.ssh_port), 
                    'connect_kwargs': {
                        'allow_agent': False,
                        'look_for_keys': False,
                        'key_filename': ['/root/.ssh/vast_rsa']}})
    
def ssh_command(label=-1):
    s = status(label)
    print(f'SSH_AUTH_SOCK="" ssh root@{s.ssh_host} -p {s.ssh_port} -o StrictHostKeyChecking=no -i /root/.ssh/vast_rsa')
