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

def _fetch(name, machine):
    from subprocess import Popen, PIPE
    from pathlib import Path

    source = str(Path(machine.root) / name / 'output')
    conn = machine.connection
    [keyfile] = conn['connect_kwargs']['key_filename']
    ssh = f"ssh -o StrictHostKeyChecking=no -i '{keyfile}' -p {conn['port']}"

    # https://unix.stackexchange.com/questions/104618/how-to-rsync-over-ssh-when-directory-names-have-spaces
    command = f"""rsync -r -e "{ssh}" {conn['user']}@{conn['host']}:"'{source}/'" test/"""
    return Popen(command, shell=True, stdout=PIPE, stderr=PIPE)

def fetch():
    from jittens.machines import machines
    from jittens.jobs import jobs
    from time import sleep

    machines = machines()
    ps = {}
    for name, job in jobs().items():
        ps[name] = _fetch(name, machines[job.machine])

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
         
    
def ssh_command(label=-1):
    s = status(label)
    print(f'SSH_AUTH_SOCK="" ssh root@{s.ssh_host} -p {s.ssh_port} -o StrictHostKeyChecking=no -i /root/.ssh/vast_rsa')
