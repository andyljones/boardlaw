from logging import getLogger
import jittens
from .api import launch, status, offers, wait, destroy
import shutil

log = getLogger(__name__)

def jittenate(local=False):
    jittens.machines.clear()
    for name, row in status().iterrows():
        if row.actual_status == 'running':
            jittens.ssh.add(name,
                resources={
                    'gpu': row.num_gpus,
                    'memory': row.cpu_ram*row.gpu_frac},
                root='/code',
                connection_kwargs={
                    'host': row.ssh_host, 
                    'user': 'root', 
                    'port': int(row.ssh_port), 
                    'connect_kwargs': {
                        'allow_agent': False,
                        'look_for_keys': False,
                        'key_filename': ['/root/.ssh/vast_rsa']}})

    if local:
        jittens.local.add(root='.jittens/local', resources={'gpu': 2})

def tails(path, jobglob='*', count=5):
    import jittens
    from pathlib import Path
    from shlex import quote
    from fnmatch import fnmatch

    machines = jittens.machines.machines()
    promises = {}
    for name, job in jittens.jobs.jobs().items():
        if fnmatch(job.name, jobglob) and job.status in ('active', 'dead'):
            if job.machine in machines:
                machine = machines[job.machine]
                dir = str(Path(machine.root) / name) + '/'
                # Split the dir and the path so we can the path being a glob, which'll fail if it's quoted
                promises[name] = machine.run(f'tail -n {count} {quote(dir)}/{path}', hide='both', asynchronous=True)
    
    stdouts = {}
    for name, promise in promises.items():
        try:
            stdouts[name] = promise.join().stdout.splitlines()[-count:]
        except Exception as e:
            stdouts[name] = ['Fabric error:'] + str(e).splitlines()[-count:]
    
    for name, stdout in stdouts.items():
        print(f'{name}:')
        for line in stdout:
            print(f'\t{line}')

         
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
