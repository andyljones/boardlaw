from logging import getLogger
from .api import launch, status, offers, wait, destroy

log = getLogger(__name__)

def jittenate(local=False, ssh_accept=True):
    import jittens
    jittens.machines.clear()
    if local:
        jittens.local.add(root='.jittens/local', resources={'gpu': 2})

    for name, row in status().iterrows():
        if row.actual_status == 'running':
            jittens.ssh.add(name,
                resources={
                    'gpu': row.num_gpus if ssh_accept else 0,
                    'memory': row.cpu_ram*row.gpu_frac/1e3},
                root='/code',
                connection_kwargs={
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

def push_command(label, source, target):
    from jittens.machines import machines
    machine = machines()[label]

    conn = machine.connection
    [keyfile] = conn['connect_kwargs']['key_filename']
    ssh = f"ssh -o StrictHostKeyChecking=no -i '{keyfile}' -p {conn['port']}"
    print(f"""rsync -r -P -e "{ssh}" "{source}" {conn['user']}@{conn['host']}:"'{target}'" """)

def gather_stats():
    from datetime import datetime
    from pathlib import Path
    import json
    count = offers('cuda_max_good >= 11.1 & gpu_name == "RTX 2080 Ti"').groupby('machine_id').num_gpus.max().sum()

    log.info(f'Count: {count}')

    path = Path('output/vast-stats.json')
    if not path.exists():
        path.write_text('[]')
    db = json.loads(path.read_text())
    db.append({'date': datetime.now().strftime(r'%Y-%m-%d %H%M%S'), 'count': count})

