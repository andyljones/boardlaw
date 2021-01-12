from . import files
from logging import getLogger
from tempfile import NamedTemporaryFile
from subprocess import CalledProcessError, check_output, STDOUT
import tarfile

log = getLogger(__name__)

def archive(run=-1):
    with NamedTemporaryFile() as f:
        # Ignores .gitignore automagically, and doesn't depend on a git repo existing
        # so that we can use it on remote machines we've rsync'd to. Hooray!
        try:
            # /dev/null fixes this bug: https://github.com/ggreer/the_silver_searcher/issues/943#issuecomment-426096765
            check_output('cd /code && ag -g "" -l -0 . </dev/null | xargs -0 tar -czvf ' + f.name, shell=True, stderr=STDOUT)
        except CalledProcessError as e:
            log.error(f'Archival failed with output "{e.stdout.decode()}"')
            raise 
        contents = f.read()

    path = files.new_file(run, 'archive.tar.gz')
    path.write_bytes(contents)

def update():
    from pavlov import runs
    import git

    rows = runs.pandas().query('tag.notnull()')
    repo = git.Repo('.')
    for run, row in rows.iterrows():
        print(run)
        if not files.path(run, 'archive.tar.gz').exists():
            repo.git.checkout(f'tags/pavlov_{row.tag}')
            archive(run)
            
    for run, row in rows.iterrows():
        with runs.update(run) as i:
            del i['tag']

def view(run=-1, path=''):
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    from IPython import display

    tarpath = files.path(run, 'archive.tar.gz')
    with tarfile.open(tarpath, 'r:gz') as f:
        if path not in f.getnames():
            paths = '\n'.join(f.getnames())
            raise IOError(f'Coudln\'t find "{path}". Options are:\n{paths}')
        contents = f.extractfile(path).read().decode()

    formatter = HtmlFormatter()
    highlighted = highlight(contents, PythonLexer(), formatter)
    display.display(display.HTML(f'<style type="text/css">{formatter.get_style_defs(".highlight")}</style>    {highlighted}'))