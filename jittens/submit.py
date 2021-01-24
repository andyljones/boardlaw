from datetime import datetime
from aljpy import humanhash
from . import state
from subprocess import STDOUT, check_output, CalledProcessError
from logging import getLogger
from pathlib import Path
from dataclasses import asdict

log = getLogger(__name__)

def compress(source, target):
    target = Path(target).absolute()
    target.parent.mkdir(exist_ok=True, parents=True)
    # ag ignores .gitignore automagically, and doesn't depend on a git repo existing
    # so that we can use it on remote machines we've rsync'd to. Hooray!
    try:
        # /dev/null fixes this bug: https://github.com/ggreer/the_silver_searcher/issues/943#issuecomment-426096765
        check_output(f'cd "{source}" && ag -g "" -l -0 . </dev/null | xargs -0 tar -czvf "{target}"', shell=True, stderr=STDOUT)
        return str(target)
    except CalledProcessError as e:
        log.error(f'Archival failed with output "{e.stdout.decode()}"')
        raise 

def submit(cmd, dir=None, **kwargs):
    now = datetime.utcnow()
    name = f'{now.strftime(r"%Y-%m-%d %H-%M-%S")} {humanhash(n=2)}'

    if dir is None:
        archive = None
    else:
        archive = compress(dir, state.ROOT / 'archives' / f'{name}.tar.gz')
    
    with state.update() as s:
        job = state.Job(
            name=name,
            submitted=str(now),
            command=cmd,
            archive=archive,
            **kwargs,
            status='fresh')
        s['jobs'][name] = asdict(job)

    return name

### TESTS

