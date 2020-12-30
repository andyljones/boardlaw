# Build and push image to DockerHub (might have to do this manually)
# Find an appropriate vast machine
# Create a machine using their commandline
# Pass onstart script to start the job
# Use vast, ssh, rsync to monitor things (using fabric and patchwork?)
# Manual destroy
import json
from subprocess import check_output
from pathlib import Path

def set_key():
    target = Path('~/.vast_api_key').expanduser()
    if not target.exists():
        key = json.loads(Path('credentials.yml').read_text())['vast']
        target.write_text(key)

def invoke(command, **kwargs):
    set_key()
    args = [command]
    for k, v in kwargs.items():
        args.append(f'--{k}')
        args.append(v)
    return check_output(args)

def suggest():
    pass