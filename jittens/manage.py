from . import jobs, machines
from logging import getLogger
from pathlib import Path

log = getLogger(__name__)

def decrement(job, machine):
    for k in set(job.resources) & set(machine.resources):
        machine.resources[k] -= job.resources[k]

def available():
    ms = machines.machines()
    for job in jobs.jobs('active').values():
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
    with jobs.update() as js:
        job = js[j.name]
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

def check_stalled():
    for job in jobs.jobs('fresh').values():
        ms = machines.machines()
        machine = select(job, ms)
        if not machine:
            log.info('Job "{job.name}" requires too many resources to run on any existing machine')

def manage():
    # Get the jobs
    for job in jobs.jobs('fresh').values():
        ms = available()
        machine = select(job, ms)
        if machine:
            launch(job, machine)

    for job in jobs.jobs('active').values():
        if dead(job):
            with jobs.update() as js:
                job = js[job.name]
                job['status'] = 'dead'

    check_stalled()

def finished():
    return all(j.status == 'dead' for j in jobs.jobs().values())

def cleanup():
    for job in jobs.jobs('dead').values():
        machines.cleanup(job)
        if job.archive:
            Path(job.archive).unlink()
        with jobs.update() as js:
            del js[job.name]

