from datetime import datetime
from aljpy import humanhash
from . import state
from subprocess import STDOUT, check_output, CalledProcessError
from logging import getLogger
from pathlib import Path

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

def submit(command, dir=None, reqs={}):
    now = datetime.utcnow()
    name = f'{now.strftime(r"%Y-%m-%d %H-%M-%S")} {humanhash(n=2)}'

    if dir is None:
        archive = None
    else:
        archive = compress(dir, state.ROOT.joinpath(name).with_suffix('.tar.gz'))
    
    spec = {
        'name': name,
        'submitted': str(now),
        'command': command,
        'archive': archive,
        'reqs': reqs,
        'attempts': []}
    with state.update() as s:
        s['submissions'].append(spec)

def submissions():
    return state.state()['submissions']

### TESTS

@state.mock_dir
def test_submission():
    submit('test')
    assert len(submissions()) == 1

@state.mock_dir
def test_compress():
    import tarfile

    p = state.ROOT / 'kitten-test-tmp'
    p.mkdir(parents=True)
    p.joinpath('test.txt').touch()

    submit('test', dir=p)

    [sub] = submissions()
    with tarfile.open(sub['archive']) as f:
        assert f.getnames() == ['test.txt']