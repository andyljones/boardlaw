import importlib
from logging import getLogger
from . import jobs
import json
import yaml
import shutil
from dataclasses import dataclass, asdict, field
from typing import List, Dict

#TODO: Config should be a superclass of Machine
@dataclass
class Machine:
    name: str
    root: str
    resources: Dict[str, int]

    def run(self, command, **kwargs):
        raise NotImplementedError()

log = getLogger(__name__)

def add(name, **kwargs):
    path = jobs.ROOT / f'machines/{name}.json'
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(kwargs))

def remove(name):
    path = jobs.ROOT / f'machines/{name}.json'
    path.unlink()

def clear():
    path = (jobs.ROOT / 'machines')
    if path.exists():
        shutil.rmtree(path)

def module(x):
    module = f'{__package__}.{x["type"]}'
    return importlib.import_module(module)

def machines() -> Dict[str, Machine]:
    machines = {}
    for path in jobs.ROOT.joinpath('machines').iterdir():
        if path.suffix in ('.json',):
            config = json.loads(path.read_text())
        elif path.suffix in ('.yaml', '.yml'):
            config = yaml.safe_load(path.read_text())
        else:
            raise IOError(f'Can\'t handle type of config file "{path}"')
            
        name = path.with_suffix('').name
        machine = module(config).Machine.create(name=name, **config)
        machines[name] = machine

    return machines

def cleanup(job):
    ms = machines()
    if job.machine not in ms:
        log.info(f'No cleanup for job {job.name} as machine "{job.machine}" no longer exists')
        return 

    machine = ms[job.machine]
    machine.cleanup(job)
