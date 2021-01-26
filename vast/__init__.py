from logging import getLogger
import jittens
from .api import launch, status, offers, wait, destroy
import shutil

log = getLogger(__name__)

def jittenate(local=False):
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

    if local:
        jittens.local.add(root='.jittens/local', resources={'gpu': 2})

def _fetch(name, machine, source, target):
    from subprocess import Popen, PIPE
    from pathlib import Path

    source = str(Path(machine.root) / name / source)
    if hasattr(machine, 'connection'):
        conn = machine.connection
        [keyfile] = conn['connect_kwargs']['key_filename']
        ssh = f"ssh -o StrictHostKeyChecking=no -i '{keyfile}' -p {conn['port']}"

        # https://unix.stackexchange.com/questions/104618/how-to-rsync-over-ssh-when-directory-names-have-spaces
        command = f"""rsync -r -e "{ssh}" {conn['user']}@{conn['host']}:"'{source}/'" "{target}" """
    else:
        command = f"""rsync -r "{source}/" "{target}" """
    return Popen(command, shell=True, stdout=PIPE, stderr=PIPE)

def fetch(source, target):
    from jittens.machines import machines
    from jittens.jobs import jobs
    from time import sleep

    machines = machines()
    ps = {}
    for name, job in jobs().items():
        if job.status in ('active', 'dead'):
            if job.machine in machines:
                ps[name] = _fetch(name, machines[job.machine], source, target)
            else:
                log.info(f'Skipping "{name}" as the machine "{job.machine}" is no longer available')

    while ps:
        for name in list(ps):
            r = ps[name].poll()
            if r is None:
                pass
            else:
                if r == 0:
                    log.debug(f'Fetched "{name}"')
                else:
                    s = ps[name].stderr.read().decode()
                    lines = '\n'.join([f'\t{l}' for l in s.splitlines()])
                    log.warn(f'Fetching "{name}" failed with retcode {r}. Stdout:\n{lines}')
                del ps[name]

        sleep(1)

def tails(path, count=50):
    from jittens.machines import machines
    from jittens.jobs import jobs
    from jittens.ssh import connection
    from pathlib import Path

    machines = machines()
    promises = {}
    stdouts = {}
    for name, job in jobs().items():
        if job.status in ('active', 'dead'):
            if job.machine in machines:
                machine = machines[job.machine]
                fullpath = Path(machine.root) / name / path
                if hasattr(machine, 'connection'):
                    conn = connection(machine)
                    promises[name] = conn.run(f'tail -n {count} "{fullpath}"', hide='both', asynchronous=True)
                else:
                    stdouts[name] = '\n'.join(fullpath.read_text().splitlines()[-count:])
    
    for name, promise in promises.items():
        try:
            stdouts[name] = promise.join().stdout
        except Exception as e:
            stdouts[name] = f'Fabric error: {e}'
    
    return stdouts

         
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
