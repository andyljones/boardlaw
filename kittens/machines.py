import importlib
from logging import getLogger
from . import state
import json
import yaml

log = getLogger(__name__)

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

def module(t):
    return importlib.import_module(t, __package__)

def machines():
    ms = {}
    for c in config():
        m = module(c['type']).machine(c)
        ms[m['name']] = m
    return ms

def launch(j, m):
    return module(m['type']).launch(j, m)

def cleanup(j):
    ms = machines()
    if j['machine'] not in ms:
        log.info(f'No cleanup for job {j["name"]} as machine "{j["machine"]}" no longer exists')
        return 

    m = ms[j['machine']]
    module(m['type']).cleanup(j, m)
