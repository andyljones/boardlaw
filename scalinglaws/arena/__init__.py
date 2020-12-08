import pandas as pd
import numpy as np
import torch
from rebar import storing, logging, dotdict
import pickle
from . import database, matchups
from .. import mohex
import activelo
import time
from logging import getLogger
from contextlib import contextmanager
from functools import wraps
from multiprocessing import Process, Event, set_start_method

log = getLogger(__name__)

def assemble_agent(agentfunc, sd, device='cpu'):
    agent = agentfunc(device=device)
    sd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sd['agent'].items()}
    agent.load_state_dict(sd)
    return agent

def periodic_agents(run_name, agentfunc, **kwargs):
    if not isinstance(run_name, (int, str)):
        agents = {}
        for r in run_name:
            agents.update(periodic_agents(r, agentfunc, **kwargs))
        return agents

    try:
        stored = storing.stored_periodic(run_name)
    except ValueError:
        return {}
    else:
        agents = {} 
        for _, row in stored.iterrows():
            name = row.date.strftime(r'%y%m%d-%H%M%S-periodic')
            sd = pickle.load(row.path.open('rb'))
            agents[name] = assemble_agent(agentfunc, sd, **kwargs)
        return agents

def latest_agent(run_name, agentfunc, **kwargs):
    try:
        sd, modified = storing.load_latest(run_name, return_modtime=True)
        return {f'{modified:%y%m%d-%H%M%S}-latest': assemble_agent(agentfunc, sd, **kwargs)}
    except ValueError:
        return {}

def suggest_periodic(n, w, G):
    valid = n.index[n.index.str.endswith('periodic')]
    n = n.reindex(index=valid, columns=valid)
    w = w.reindex(index=valid, columns=valid)
    try:
        soln = activelo.solve(n.values, w.values)
        log.info(f'Fitted a posterior, {(soln.σd**2).mean()**.5:.2f}σd over {n.shape[0]} agents')
        matchup = activelo.suggest(soln, G)
    except ValueError:
        log.warn('Solver failed; making a random suggestion')
        matchup = tuple(np.random.randint(0, n.shape[0], (2,)))
    return [n.index[s] for s in matchup]

def suggest_latest(n, w):
    if n.index.endswith('latest').any() and n.index.endswith('periodic').any():
        [latest] = n.index[n.index.endswith('latest')]
        periodic = n.index[n.index.endswith('periodic')][-1]
        return (latest, periodic)

def suggest_mohex(n, w, G):
    valid = n.index[n.index.str.endswith('periodic') | (n.index == 'mohex')]
    n = n.reindex(index=valid, columns=valid)
    w = w.reindex(index=valid, columns=valid)
    try:
        soln = activelo.solve(n.values, w.values)
        log.info(f'Fitted a posterior, {(soln.σd**2).mean()**.5:.2f}σd over {n.shape[0]} agents')
        improvement = activelo.improvement(soln, G)
        improvement = pd.DataFrame(improvement, n.index, n.columns).loc['mohex'].drop('mohex')
        return ('mohex', improvement.argmax())
    except ValueError:
        log.warn('Solver failed; making a random suggestion')
        return ('mohex', np.random.randint(0, n.shape[0]))

def step(run_name, worlds, agents, kind):
    if len(agents) < 2:
        log.info(f'Only {len(agents)} agents have been loaded')
        return 

    log.info(f'Running a {kind} game')
    games, wins = database.symmetric_pandas(run_name, agents)
    log.info(f'Loaded {int(games.sum().sum())} games')
    if kind == 'periodic':
        matchup = suggest_periodic(games, wins, worlds.periodic.n_envs)
    elif kind == 'latest':
        matchup = suggest_latest(games, wins)
    elif kind == 'mohex':
        matchup = suggest_mohex(games, wins, worlds.mohex.n_envs)
    else:
        raise ValueError(f'Kind "{kind}" is invalid')

    if matchup:
        agents = {m: agents[m] for m in matchup}
        log.info('Playing ' + ' v. '.join(agents))
        results = matchups.evaluate(worlds[kind], agents)

        if kind in ('periodic', 'mohex'):
            database.store(run_name, results)
            log.info('Stored')

def arena(run_name, worldfunc, agentfunc, device='cpu', **kwargs):
    with logging.via_dir(run_name):
        worlds = dotdict.dotdict(
            periodic=worldfunc(device=device, n_envs=256),
            latest=worldfunc(device=device, n_envs=256),
            mohex=worldfunc(device=device, n_envs=8))

        mhx = mohex.MoHexAgent()
        kinds = ('periodic', 'latest', 'mohex')
        
        i = 0
        agents = {}
        last_load, last_step = 0, 0
        while True:
            if time.time() - last_load > 60:
                last_load = time.time()
                agents = {
                    **periodic_agents(run_name, agentfunc, device=device),
                    **latest_agent(run_name, agentfunc, device=device),
                    'mohex': mhx}
            
            if time.time() - last_step > 1:
                last_step = time.time()
                step(run_name, worlds, agents, kinds[i % len(kinds)])
                i += 1

@wraps(arena)
@contextmanager
def monitor(*args, **kwargs):
    set_start_method('spawn', True)
    p = Process(target=arena, args=args, kwargs=kwargs, name='arena-monitor')
    try:
        p.start()
        yield
    finally:
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
    arena('test', worldfunc, agentfunc, ref_runs=['2020-11-27 19-40-27 az-test'])