from logging import getLogger
from . import state
import json
import yaml

log = getLogger(__name__)

TYPES = {}

def register(cls):
    TYPES[cls.__name__.lower()] = cls
    return cls

def config():
    configs = []
    for path in state.ROOT.joinpath('machines').iterdir():
        if path.suffix in ('.json',):
            content = json.loads(path.read_text())
        elif path.suffix in ('.yaml', '.yml'):
            content = yaml.load(path.read_text())
        else:
            content = []
            log.warn(f'Can\'t handle type of config file "{path}"')
        
        if isinstance(content, dict):
            content = [content]

        configs.extend(content)

    return configs

def machines():
    ms = {}
    for c in config():
        m = TYPES[c['type']].machine(c)
        ms[m['name']] = m
    return ms

def launch(j, m):
    return TYPES[m['type']].launch(j, m)

def cleanup(j):
    ms = machines()
    if j['machine'] not in ms:
        log.info(f'No cleanup for job {j["name"]} as machine "{j["machine"]}" no longer exists')
        return 

    m = ms[j['machine']]
    TYPES[m['type']].cleanup(j, m)
