import pandas as pd
import numpy as np
import torch
from rebar import storing, logging, dotdict, stats
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

def step_periodic(run_name, worlds, agents):
    n, w = database.symmetric_pandas(run_name, agents)
    log.info(f'Loaded {int(n.sum().sum())} games')

    valid = n.index[n.index.str.endswith('periodic')]
    n = n.reindex(index=valid, columns=valid)
    w = w.reindex(index=valid, columns=valid)

    soln = activelo.solve(n.values, w.values)
    log.info(f'Fitted a posterior, {(soln.σd**2).mean()**.5:.2f}σd over {n.shape[0]} agents')
    matchup = activelo.suggest(soln, worlds.periodic.n_envs)
    matchup = [n.index[m] for m in matchup]

    agents = {m: agents[m] for m in matchup}
    log.info('Playing ' + ' v. '.join(agents))
    results = matchups.evaluate(worlds.periodic, agents)

    wins, games = int(results[0].wins[0] + results[1].wins[1]), int(sum(r.games for r in results))
    log.info(f'Storing. {wins} wins in {games} games for {list(agents)[0]} ')
    database.store(run_name, results)

def step_mohex(run_name, worlds, agents):
    n, w = database.symmetric_pandas(run_name, agents)
    log.info(f'Loaded {int(n.sum().sum())} games')

    valid = n.index[n.index.str.endswith('periodic') | (n.index == 'mohex')]
    n = n.reindex(index=valid, columns=valid)
    w = w.reindex(index=valid, columns=valid)

    soln = activelo.solve(n.values, w.values)
    log.info(f'Fitted a posterior, {(soln.σd**2).mean()**.5:.2f}σd over {n.shape[0]} agents')
    improvement = activelo.improvement(soln, worlds.mohex.n_envs)
    improvement = pd.DataFrame(improvement, n.index, n.columns).loc['mohex'].drop('mohex')
    matchup = ('mohex', improvement.idxmax())

    agents = {m: agents[m] for m in matchup}
    log.info('Playing ' + ' v. '.join(agents))
    results = matchups.evaluate(worlds.mohex, agents)

    wins, games = int(results[0].wins[0] + results[1].wins[1]), int(sum(r.games for r in results))
    log.info(f'Storing. {wins} wins in {games} games for {list(agents)[0]} ')
    database.store(run_name, results)

def step_latest(run_name, worlds, agents):
    n, w = database.symmetric_pandas(run_name, agents)
    log.info(f'Loaded {int(n.sum().sum())} games')

    [latest] = n.index[n.index.str.endswith('latest')]
    periodic = n.index[n.index.str.endswith('periodic')][-1]
    matchup = (latest, periodic)

    agents = {m: agents[m] for m in matchup}
    log.info('Playing ' + ' v. '.join(agents))
    results = matchups.evaluate(worlds.periodic, agents)

    for r in results:
        w.loc[r.names[0], r.names[1]] += r.wins[0]
        w.loc[r.names[1], r.names[0]] += r.wins[1]
        n.loc[r.names[0], r.names[1]] += r.games
    
    wins, games = int(results[0].wins[0] + results[1].wins[1]), int(sum(r.games for r in results))
    log.info(f'Fitting posterior. {wins} wins for {list(agents)[0]} in {games} games')
    soln = activelo.solve(n.values, w.values)
    log.info(f'Fitted posterior, {(soln.σd**2).mean()**.5:.2f}σd over {n.shape[0]} agents')
    μ = pd.Series(soln.μ, n.index).loc[latest]
    log.info(f'eElo for {latest} is approximately {μ:.2f}')
    stats.mean('elo', μ)

def step(run_name, worlds, agents, kind):
    if len(agents) < 2:
        log.info(f'Only {len(agents)} agents have been loaded')
        return 

    log.info(f'Running a "{kind}" step')
    try:
        globals()[f'step_{kind}'](run_name, worlds, agents)
    except Exception as e:
        log.error(f'Failed while running a "{kind}" step with a {e} error')

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