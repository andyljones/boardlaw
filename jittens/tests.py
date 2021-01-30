from pathlib import Path
import shutil
from . import jobs, manage, finished, cleanup, local, ssh
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

        name = jobs.submit('python test.py >logs.txt', dir=d, resources={'gpu': 1})

    archive = jobs.ROOT / 'archives' / f'{name}.tar.gz'
    assert archive.exists()

    while not finished():
        manage()

    import time
    time.sleep(1)

    dir = jobs.ROOT / 'local' / name
    assert (dir / 'test.py').exists()
    assert (dir / 'logs.txt').exists()
    assert (dir / 'logs.txt').read_text() == '0\n'

    cleanup()

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
            cmd='python test.py >logs.txt', 
            dir=d, 
            resources={'gpu': 1})

    archive = jobs.ROOT / 'archives' / f'{name}.tar.gz'
    assert archive.exists()

    while not finished():
        manage()

    dir = jobs.ROOT / 'ssh' / name
    assert (dir / 'test.py').exists()
    assert (dir / 'logs.txt').exists()
    assert (dir / 'logs.txt').read_text() == '0\n'


DEMO = '''
import os
import time
time.sleep(5)
print({i})'''

@mock_dir
def demo():
    # Add a faux local machine to jittens
    # This just writes out a json file in `.jittens/machines`
    mock_local_config()

    # Add a faux SSH machine too
    # This just writes out another json file in `.jittens/machines`
    mock_ssh_config()

    with TemporaryDirectory() as d:
        for i in range(5):
            # Write out five different scripts to run as jobs
            (Path(d) / 'demo.py').write_text(DEMO.format(i=i))

            # Submit them all, asking for one GPU each.
            # This'll package up the dir into an archive and add an entry
            # to the `.jittens/jobs.json`
            jobs.submit(
                'python demo.py >logs.txt', 
                dir=d, 
                resources={'gpu': 1})

    # While there are jobs left alive, keep `manage`ing
    while not finished():
        # This looks over the `jobs.json`, and if any fit in the available 
        # machines as described by the `machines/` json files, it copies over 
        # the archive, sets the job running, and gets a PID back. PID goes 
        # into the job's entry in the `jobs.json`, and in every call after that, 
        # if the PID's disappeared the job is marked dead.
        manage()
