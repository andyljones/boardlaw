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
    log.info(f'There are {len(jobs.jobs("active"))} active jobs and {len(jobs.jobs("fresh"))} fresh jobs.')

    # See if any of the active jobs are now dead
    for job in jobs.jobs('active').values():
        if dead(job, ms):
            job.status = 'dead'
            with jobs.update() as js:
                js[job.name] = asdict(job)

    # See if any fresh jobs can be submitted
    ms = machines.machines()
    for job in jobs.jobs('fresh').values():
        av = available(ms)
        machine = select(job, av)
        if machine:
            launch(job, machine)

    check_stalled(ms)

def finished():
    return all(j.status == 'dead' for j in jobs.jobs().values())

def cleanup(names=None):
    for job in jobs.jobs('dead').values():
        if names is not None and job.name in names:
            log.info(f'Cleaning up {job.name}')
            machines.cleanup(job)
            if job.archive:
                Path(job.archive).unlink()
            with jobs.update() as js:
                del js[job.name]

def fetch(source, target):
    from jittens.machines import machines
    from jittens.jobs import jobs

    machines = machines()
    ps = {}
    queue = iter(jobs().items())
    fetched = []
    while True:
        # Anything more than 10 and default SSH configs start having trouble, throwing 235 & 255 errors.
        # Need to up `MaxStartups` if you wanna go higher.
        if len(ps) <= 1:
            try:
                name, job = next(queue)
                if job.status in ('active', 'dead'):
                    if job.machine in machines:
                        log.info(f'Fetching "{name}" from "{job.machine}"')
                        ps[name] = machines[job.machine].fetch(job.name, source, target)
                    else:
                        log.info(f'Skipping "{name}" as the machine "{job.machine}" is no longer available')
            except StopIteration:
                pass

        if not ps:
            break

        for name in list(ps):
            try:
                r = ps[name].join()
            except Exception as e:
                log.info(f'Failed to fetch {name}: {e}')
            else:
                log.debug(f'Fetched "{name}"')
                fetched.append(name)
            del ps[name]

    return fetched

def tails(path, jobglob='*', count=5):
    from pathlib import Path
    from shlex import quote
    from fnmatch import fnmatch
    from . import machines

    ms = machines.machines()
    promises = {}
    for name, job in jobs.jobs().items():
        if fnmatch(job.name, jobglob) and job.status in ('active', 'dead'):
            if job.machine in ms:
                machine = ms[job.machine]
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

         