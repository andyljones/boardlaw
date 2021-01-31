from . import jobs, machines
from dataclasses import asdict
from logging import getLogger
from pathlib import Path
import copy

log = getLogger(__name__)

def decrement(job, machine):
    for k in set(job.allocation) & set(machine.resources):
        machine.resources[k] = list(set(machine.resources[k]) - set(job.allocation[k]))

def available(ms):
    ms = copy.deepcopy(ms)
    for job in jobs.jobs('active').values():
        if job.machine in ms:
            decrement(job, ms[job.machine])
    return ms

def viable(asked, offered):
    for k in asked:
        if k not in offered:
            return False
        if asked[k] > len(offered[k]):
            return False
    return True

def select(job, machines):
    for m in machines.values():
        if viable(job.resources, m.resources):
            return m

def allocate(job, machine):
    alloc = {}
    for k, count in job.resources.items():
        alloc[k] = machine.resources[k][:count]
    return alloc

def launch(job, machine):
    log.info(f'Launching job "{job.name}" on machine "{machine.name}"')
    allocation = allocate(job, machine)
    job.status = 'active'
    job.machine = machine.name
    job.process = machine.launch(job, allocation)
    job.allocation = allocation
    with jobs.update() as js:
        js[job.name] = asdict(job)
    log.info(f'Launched with PID #{job.process}')

def dead(job, ms):
    if job.machine not in ms:
        log.info(f'Job "{job.name}" has died as the machine "{job.machine}" no longer exists')
        return True
    if job.process not in ms[job.machine].processes:
        log.info(f'Job "{job.name}" has died as its PID #{job.process} is not visible on "{job.machine}"')
        return True
    return False

def check_stalled(ms):
    for job in jobs.jobs('fresh').values():
        machine = select(job, ms)
        if not machine:
            log.info(f'Job "{job.name}" requires too many resources to run on any existing machine')

def refresh(ms=None):
    ms = machines.machines() if ms is None else ms

    # See if any of the active jobs are now dead
    log.info(f'There are {len(jobs.jobs("active"))} active jobs.')
    for job in jobs.jobs('active').values():
        if dead(job, ms):
            job.status = 'dead'
            with jobs.update() as js:
                js[job.name] = asdict(job)

    # See if any fresh jobs can be submitted
    log.info(f'There are {len(jobs.jobs("fresh"))} fresh jobs.')
    ms = machines.machines()
    for job in jobs.jobs('fresh').values():
        av = available(ms)
        machine = select(job, av)
        if machine:
            launch(job, machine)

    check_stalled(ms)

def finished():
    return all(j.status == 'dead' for j in jobs.jobs().values())

def cleanup():
    for job in jobs.jobs('dead').values():
        log.info(f'Cleaning up {job.name}')
        machines.cleanup(job)
        if job.archive:
            Path(job.archive).unlink()
        with jobs.update() as js:
            del js[job.name]

def kill(name):
    pass
    