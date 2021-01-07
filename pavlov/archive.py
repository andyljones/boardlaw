from . import files
from logging import getLogger
from tempfile import NamedTemporaryFile
from subprocess import check_output, STDOUT


log = getLogger(__name__)

def archive(run=-1):
    with NamedTemporaryFile() as f:
        check_output('git add -u', shell=True)
        check_output('git ls-files --cached --others --exclude-standard -z | xargs -0 tar -czvf ' + f.name, shell=True, stderr=STDOUT)
        contents = f.read()

    path = files.new_file(run, 'archive.tar.gz')
    path.write_bytes(contents)