import importlib
from logging import getLogger
from . import state
import json
import yaml
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Machine:
    name: str
    resources: Dict[str, int]
    root: str
    processes: List[int]


log = getLogger(__name__)

def configurations():
    configs = []
    for path in state.ROOT.joinpath('machines').iterdir():
        if path.suffix in ('.json',):
            content = json.loads(path.read_text())
        elif path.suffix in ('.yaml', '.yml'):
            content = yaml.safe_load(path.read_text())
        else:
            content = []
            log.warn(f'Can\'t handle type of config file "{path}"')
        
        if isinstance(content, dict):
            content = [content]

        configs.extend(content)

    return configs

def write(name, configs):
    path = state.ROOT / f'machines/{name}.json'
    path.write_text(json.dumps(configs))

def module(config):
    return importlib.import_module(f'{__package__}.{config["type"]}')

def machines() -> Dict[str, Machine]:
    ms = {}
    for config in configurations():
        m = module(config).machine(config)
        ms[m['name']] = m
    return ms

def launch(job, machine) -> int:
    return module(machine).launch(job, machine)

def cleanup(job):
    ms = machines()
    if job['machine'] not in ms:
        log.info(f'No cleanup for job {job["name"]} as machine "{job["machine"]}" no longer exists')
        return 

    machine = ms[job['machine']]
    module(machine).cleanup(job, machine)
