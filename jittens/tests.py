import json
from pathlib import Path
import shutil
from . import jobs, manage, local, ssh
from tempfile import TemporaryDirectory

def mock_dir(f):
    def g(*args, **kwargs):
        try:
            OLD = jobs.ROOT
            jobs.ROOT = Path('.jittens-test')
            if jobs.ROOT.exists():
                shutil.rmtree(jobs.ROOT)
            return f(*args, **kwargs)
        finally:
            jobs.ROOT = OLD
    return g

### Test submit

@mock_dir
def test_submission():
    jobs.submit('test')
    assert len(jobs.jobs()) == 1

@mock_dir
def test_compress():
    import tarfile

    p = jobs.ROOT / 'kitten-test-tmp'
    p.mkdir(parents=True)
    p.joinpath('test.txt').touch()

    jobs.submit('test', dir=p)

    [job] = jobs.jobs()
    with tarfile.open(job.archive) as f:
        assert f.getnames() == ['test.txt']

def mock_local_config():
    local.add(
        root=str(jobs.ROOT / 'local'),
        resources={'gpu': 2, 'memory': 64})

### Test local

@mock_dir
def test_local():

    mock_local_config()

    with TemporaryDirectory() as d:
        script = Path(d) / 'test.py'
        script.write_text('import os; print(os.environ["JITTENS_GPU"])')

        name = jobs.submit(
            'python test.py', dir=d, 
            resources={'gpu': 1}, stdout='logs.txt', stderr='logs.txt')

    archive = jobs.ROOT / 'archives' / f'{name}.tar.gz'
    assert archive.exists()

    while not manage.finished():
        manage.manage()

    dir = jobs.ROOT / 'local' / name
    assert (dir / 'test.py').exists()
    assert (dir / 'logs.txt').exists()
    assert (dir / 'logs.txt').read_text() == '1:2\n'

    manage.cleanup()

    assert not dir.exists()
    assert not archive.exists()

### Test ssh

def mock_ssh_config():
    ssh.add('ssh',
        resources={
            'gpu': 2,
            'memory': 64},
        root=str((jobs.ROOT / 'ssh').absolute()),
        connection={
            'host': 'localhost', 
            'user': 'root', 
            'port': '22', 
            'connect_kwargs': {
                'allow_agent': False,
                'look_for_keys': False,
                'key_filename': ['/root/.ssh/boardlaw_rsa']}})

@mock_dir
def test_ssh():
    mock_ssh_config()

    with TemporaryDirectory() as d:
        script = Path(d) / 'test.py'
        script.write_text('import os; print(os.environ["JITTENS_GPU"])')

        name = jobs.submit(
            cmd='python test.py', 
            dir=d, 
            resources={'gpu': 1}, 
            stdout='logs.txt', 
            stderr='logs.txt')

    archive = jobs.ROOT / 'archives' / f'{name}.tar.gz'
    assert archive.exists()

    while not manage.finished():
        manage.manage()

    dir = jobs.ROOT / 'ssh' / name
    assert (dir / 'test.py').exists()
    assert (dir / 'logs.txt').exists()
    assert (dir / 'logs.txt').read_text() == '1:2\n'


DEMO = '''
import os
import time
time.sleep(5)
print({i})'''

@mock_dir
def demo():
    mock_local_config()
    mock_ssh_config()

    with TemporaryDirectory() as d:
        for i in range(5):
            (Path(d) / 'demo.py').write_text(DEMO.format(i=i))

            jobs.submit(
                'python demo.py', 
                dir=d, 
                resources={'gpu': 1}, 
                stdout='logs.txt', 
                stderr='logs.txt')

    while not manage.finished():
        manage.manage()
