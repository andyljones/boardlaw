import json
from portalocker import RLock
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict, field
from subprocess import STDOUT, check_output, CalledProcessError
from logging import getLogger
from datetime import datetime
from aljpy import humanhash
from shlex import quote

log = getLogger(__name__)

ROOT = Path('.jittens')

#TODO: Subclass this for 'Job awaiting submission' and 'Job submitted'
@dataclass
class Job:
    name: str
    submitted: str
    command: str
    resources: Dict[str, int]
    status: str
    archive: str = ''
    params: Dict[str, None] = field(default_factory=dict)

    allocation: Dict[str, List[int]] = field(default_factory=dict)
    machine: Optional[str] = None
    process: Optional[str] = None

def path():
    return ROOT / 'jobs.json'

_lock = None
@contextmanager
def lock():
    # It's tempting to lock on the state.json file. But that leads to some confusion about 
    # how to handle race conditions when *creating* the _info.json file,
    # and also about how to handle global operations that aren't exclusively
    # about that file. Better to just lock on a purpose-made lock file.

    # Re-entrancy is dealt with on the object rather than on the handle, so need to 
    # keep the object itself about. Ffff.
    global _lock
    if _lock is None:
        _lock = RLock(ROOT / '_lock')

    ROOT.mkdir(exist_ok=True, parents=True)
    with _lock:
        yield

def raw_jobs():
    with lock():
        if not path().exists():
            path().parent.mkdir(exist_ok=True, parents=True)
            path().write_text(json.dumps({}))
        return json.loads(path().read_text())

@contextmanager
def update():
    with lock():
        js = raw_jobs()
        yield js
        path().write_text(json.dumps(js))

def jobs(status=None) -> Dict[str, Job]:
    if status:
        return {name: job for name, job in jobs().items() if job.status == status}
    return {name: Job(**job) for name, job in raw_jobs().items()}

def compress(source, target, extras):
    target = Path(target).absolute()
    target.parent.mkdir(exist_ok=True, parents=True)
    # ag ignores .gitignore automagically, and doesn't depend on a git repo existing
    # so that we can use it on remote machines we've rsync'd to. Hooray!
    try:
        # /dev/null fixes this bug: https://github.com/ggreer/the_silver_searcher/issues/943#issuecomment-426096765
        check_output(f'cd {quote(source)} && ag -g "" -l -0 . </dev/null | xargs -0 tar -czvf {quote(str(target))}', shell=True, stderr=STDOUT)
        for extra in extras:
            check_output(f'cd {quote(source)} && tar rvf {quote(str(target))} {quote(extra)}')
        return str(target)
    except CalledProcessError as e:
        log.error(f'Archival failed with output "{e.stdout.decode()}"')
        raise 

def submit(cmd, dir=None, extras=[], **kwargs):
    now = datetime.utcnow()
    name = f'{now.strftime(r"%Y-%m-%d %H-%M-%S")} {humanhash(n=2)}'

    if dir is None:
        archive = ''
    else:
        archive = compress(dir, ROOT / 'archives' / f'{name}.tar.gz', extras)
    
    with update() as js:
        log.info(f'Submitting job "{name}"')
        job = Job(
            name=name,
            submitted=str(now),
            command=cmd,
            archive=archive,
            **kwargs,
            status='fresh')
        js[name] = asdict(job)

    return name

def delete(name=None):
    if name is None:
        for name in jobs():
            delete(name)
    else:
        with update() as js:
            del js[name]

