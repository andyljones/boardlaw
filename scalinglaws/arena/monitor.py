import torch
from rebar import storing, logging
import pickle
from . import database, matchups
import time
from logging import getLogger
from contextlib import contextmanager
from functools import wraps
from multiprocessing import Process, Event

log = getLogger(__name__)

def assemble_agent(agentfunc, sd, device='cpu'):
    agent = agentfunc(device=device)
    sd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sd['agent'].items()}
    agent.load_state_dict(sd)
    return agent

def periodic_agents(run_name, agentfunc):
    if not isinstance(run_name, str):
        agents = {}
        for r in run_name:
            agents.update(periodic_agents(r, agentfunc))
        return agents

    try:
        stored = storing.stored_periodic(run_name)
    except ValueError:
        return {}
    else:
        agents = {} 
        for _, row in stored.iterrows():
            name = row.date.strftime('%a-%H%M%S')
            sd = pickle.load(row.path.open('rb'))
            agents[name] = assemble_agent(agentfunc, sd)
        return agents

def latest_agent(run_name, agentfunc, **kwargs):
    sd = storing.load_latest(run_name)
    return assemble_agent(agentfunc, sd, **kwargs)

def run(run_name, worldfunc, agentfunc, device='cpu', ref_runs=[], canceller=None, **kwargs):
    with logging.via_dir(run_name):
        matcher = matchups.AdaptiveMatcher(worldfunc, device=device, **kwargs)
        runs = ref_runs + [run_name]
        
        last_load, last_step = 0, 0
        while True:
            if time.time() - last_load > 60:
                last_load = time.time()
                agents = periodic_agents(runs, agentfunc)
                matcher.add_agents(agents)
                log.info(f'Loading {len(agents)} agents')
            
            if time.time() - last_step > 1:
                last_step = time.time()
                results = matcher.step()
                database.store(run_name, results)
                log.info(f'Stepped, stored {len(results)} results')
            
            if canceller and canceller.is_set():
                log.info('Breaking')
                break

@wraps(run)
@contextmanager
def monitor(*args, **kwargs):
    canceller = Event()
    kwargs = {**kwargs, 'canceller': canceller}
    p = Process(target=run, args=args, kwargs=kwargs)
    try:
        p.start()
        yield
    finally:
        canceller.set()
        for _ in range(50):
            if not p.is_alive():
                log.info('Arena monitor dead')
                break
            time.sleep(.1)
        else:
            log.info('Abruptly terminating arena monitor; it should have shut down naturally!')
            p.terminate()

def test():
    from scalinglaws import worldfunc, agentfunc
    from rebar import paths
    paths.clear('test')
    run('test', worldfunc, agentfunc, ref_runs=['2020-11-27 19-40-27 az-test'])