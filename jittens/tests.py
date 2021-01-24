from pathlib import Path
import shutil
from . import state, manage, submit, local, ssh
from tempfile import TemporaryDirectory

def mock_dir(f):
    def g(*args, **kwargs):
        global ROOT
        try:
            OLD = state.ROOT
            ROOT = Path('.kittens-test')
            if ROOT.exists():
                shutil.rmtree(ROOT)
            return f(*args, **kwargs)
        finally:
            ROOT = OLD
    return g

def mock_local_config():
    import json
    path = state.ROOT / 'machines' / 'local.json'
    path.parent.mkdir(exist_ok=True, parents=True)

    content = json.dumps({
        'type': 'local', 
        'root': str(state.ROOT / 'local'),
        'resources': {'gpu': 2, 'memory': 64}})
    path.write_text(content)

@mock_dir
def test_local():

    mock_local_config()

    with TemporaryDirectory() as d:
        script = Path(d) / 'test.py'
        script.write_text('import os; print(os.environ["KITTENS_GPU"])')

        name = submit.submit(
            'python test.py', dir=d, 
            resources={'gpu': 1}, stdout='logs.txt', stderr='logs.txt')

    while not manage.finished():
        manage.manage()

    archive = state.ROOT / 'archives' / f'{name}.tar.gz'
    assert archive.exists()

    dir = state.ROOT / 'local' / name
    assert (dir / 'test.py').exists()
    assert (dir / 'logs.txt').exists()
    assert (dir / 'logs.txt').read_text() == '1:2\n'

    manage.cleanup()

    assert not dir.exists()
    assert not archive.exists()

@mock_dir
def test_ssh():
    pass