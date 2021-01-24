import importlib
from logging import getLogger
from . import jobs
import json
import yaml
import shutil
from dataclasses import dataclass, asdict
from typing import List, Dict

#TODO: Config should be a superclass of Machine
@dataclass
class Machine:
    name: str
    resources: Dict[str, int]
    root: str
    processes: List[int]

log = getLogger(__name__)

def add(name, **kwargs):
    path = jobs.ROOT / f'machines/{name}.json'
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(kwargs))

def remove(name):
    path = jobs.ROOT / f'machines/{name}.json'
    path.unlink()

def clear():
    shutil.rmtree(jobs.ROOT / 'machines')

def module(x):
    if isinstance(x, dict):
        module = f'{__package__}.{x["type"]}'
    elif isinstance(x, Machine):
        module = x.__module__
    else:
        raise ValueError()
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
        machine = module(config).machine(name, config)
        machines[name] = machine

    return machines

def launch(job, machine) -> int:
    return module(machine).launch(job, machine)

def cleanup(job):
    ms = machines()
    if job.machine not in ms:
        log.info(f'No cleanup for job {job.name} as machine "{job.machine}" no longer exists')
        return 

    machine = ms[job.machine]
    module(machine).cleanup(job, machine)
