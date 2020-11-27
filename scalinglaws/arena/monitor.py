from scalinglaws.heads import Tensor
import torch
import time
from rebar import storing, logging
import pickle
from . import database, matchups
from logging import getLogger
from functools import wraps
from contextlib import contextmanager
from multiprocessing import Process, Event, set_start_method

log = getLogger(__name__)

def assemble_agent(agentfunc, sd, device='cpu'):
    agent = agentfunc(device=device)
    #TODO: Should be remapping devices with torch's map_location
    sd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sd['agent'].items()}
    agent.load_state_dict(sd)
    return agent

def periodic_agents(run_name, agentfunc, device='cpu'):
    try:
        stored = storing.stored_periodic(run_name)
    except ValueError:
        return {}
    else:
        agents = {} 
        for _, row in stored.iterrows():
            name = row.date.strftime('%a-%H%M%S')
            sd = pickle.load(row.path.open('rb'))
            agents[name] = assemble_agent(agentfunc, sd, device)
        return agents

def latest_agent(run_name, agentfunc):
    sd = storing.load_latest(run_name)
    return assemble_agent(agentfunc, sd)

def run(run_name, worldfunc, agentfunc, device='cpu', canceller=None):
    matcher = matchups.AdaptiveMatcher(worldfunc, device=device)
    last_load = 0
    last_loop = 0
    with logging.via_dir(run_name):
        while True:
            if time.time() - last_loop > 1:
                last_loop = time.time()
                if time.time() - last_load > 60:
                    last_load = time.time()
                    agents = periodic_agents(run_name, agentfunc)
                    matcher.add_agents(agents)
                    log.info(f'Loaded {len(agents)} agents')
                
                results = matcher.step()
                database.store(run_name, results)
                log.info(f'Stepped, stored {len(results)} results')

                if canceller and canceller.is_set():
                    log.info('Cancelled')
                    break
            
@wraps(run)
@contextmanager
def monitor(*args, **kwargs):
    set_start_method('spawn', True)
    canceller = Event()
    kwargs = {**kwargs, 'canceller': canceller}
    p = Process(target=run, args=args, kwargs=kwargs, name='monitor')
    p.start()
    log.info('Launched')
    try:
        yield
    finally:
        log.info('Cancelling')
        canceller.set()
        for _ in range(50):
            time.sleep(.1)
            if not p.is_alive():
                break
        log.info('Dead')