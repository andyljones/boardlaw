from . import state, machines
from logging import getLogger
from pathlib import Path

log = getLogger(__name__)

def decrement(job, machine):
    for k in set(job.resources) & set(machine.resources):
        machine.resources[k] -= job.resources[k]

def available():
    ms = machines.machines()
    for job in state.jobs('active').values():
        if job.machine in ms:
            decrement(job, ms[job.machine])
    return ms

def viable(asked, offered):
    for k in asked:
        if k not in offered:
            return False
        if asked[k] > offered[k]:
            return False
    return True

def select(j, ms):
    for m in ms.values():
        if viable(j.resources, m.resources):
            return m

def launch(j, m):
    log.info(f'Launching job "{j.name}" on machine "{m.name}"')
    pid = machines.launch(j, m)
    log.info(f'Launched with PID #{pid}')
    with state.update() as s:
        job = s['jobs'][j.name]
        job['status'] = 'active'
        job['machine'] = m.name
        job['process'] = pid

def dead(job):
    ms = machines.machines()
    if job.machine not in ms:
        log.info(f'Job "{job.name}" has died as the machine "{job.machine}" no longer exists')
        return True
    if job.process not in ms[job.machine].processes:
        log.info(f'Job "{job.name}" has died as its PID #{job.process} is not visible on "{job.machine}"')
        return True
    return False

def manage():
    # Get the jobs
    for job in state.jobs('fresh').values():
        ms = available()
        machine = select(job, ms)
        if machine:
            launch(job, machine)

    for job in state.jobs('active').values():
        if dead(job):
            with state.update() as s:
                job = s['jobs'][job.name]
                job['status'] = 'dead'

def finished():
    return all(j.status == 'dead' for j in state.jobs().values())

def cleanup():
    for job in state.jobs('dead').values():
        machines.cleanup(job)
        if job.archive:
            Path(job.archive).unlink()
        with state.update() as s:
            del s['jobs'][job.name]

@state.mock_dir
def demo():
    from kittens import submit, local
    from tempfile import TemporaryDirectory

    local.mock_config()

    with TemporaryDirectory() as d:
        script = Path(d) / 'test.py'
        script.write_text('import os; print(os.environ["KITTENS_GPU"])')

        cmd = 'python test.py'
        name = submit.submit(cmd, dir=d, 
            resources={'gpu': 1}, stdout='logs.txt', stderr='logs.txt')

    while not finished():
        manage()

    archive = state.ROOT / 'archives' / f'{name}.tar.gz'
    assert archive.exists()

    dir = state.ROOT / 'local' / name
    assert (dir / 'test.py').exists()
    assert (dir / 'logs.txt').exists()
    assert (dir / 'logs.txt').read_text() == '1:2\n'

    cleanup()

    assert not dir.exists()
    assert not archive.exists()
