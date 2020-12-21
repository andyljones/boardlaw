import pandas as pd
import torch
from rebar import storing, logging, dotdict, stats, paths
from . import trials, mohex
import time
from logging import getLogger
from contextlib import contextmanager
from functools import wraps
from multiprocessing import Process, set_start_method

# Re-export
from .plot import heatmap, periodic, nontransitivities
from .analysis import elos

log = getLogger(__name__)

def assemble_agent(agentfunc, sd, device='cpu'):
    agent = agentfunc(device=device)
    sd = storing.expand_once(sd)['agent']
    agent.load_state_dict(sd)
    return agent

def periodic_agents(run_name, agentfunc, device='cpu', **kwargs):
    if not isinstance(run_name, (int, str)):
        agents = {}
        for r in run_name:
            agents.update(periodic_agents(r, agentfunc, device=device, **kwargs))
        return agents

    try:
        stored = storing.stored_periodic(run_name)
    except ValueError:
        return {}
    else:
        agents = {} 
        for _, row in stored.iterrows():
            name = row.date.strftime(r'%y%m%d-%H%M%S-periodic')
            sd = torch.load(row.path, map_location=device)
            agents[name] = assemble_agent(agentfunc, sd, device=device, **kwargs)
        return agents

def latest_agent(run_name, agentfunc, device='cpu', **kwargs):
    try:
        sd, modified = storing.load_latest(run_name, device=device, return_modtime=True)
        return {f'{modified:%y%m%d-%H%M%S}-latest': assemble_agent(agentfunc, sd, device=device, **kwargs)}
    except ValueError:
        return {}

def arena(run_name, worldfunc, agentfunc, device='cuda:1'):
    run_name = paths.resolve(run_name)
    with logging.to_dir(run_name), stats.to_dir(run_name):
        worlds = dotdict.dotdict(
            periodic=worldfunc(device=device, n_envs=256),
            latest=worldfunc(device=device, n_envs=256),
            mohex=worldfunc(device=device, n_envs=8))

        from .. import mohex
        mhx = mohex.MoHexAgent()
        kinds = list(worlds)
        
        i = 0
        agents = {}
        last_load, last_step = 0, 0
        while True:
            if time.time() - last_load > 15:
                last_load = time.time()
                agents = {
                    **periodic_agents(run_name, agentfunc, device=device),
                    **latest_agent(run_name, agentfunc, device=device),
                    'mohex': mhx}
            
            if time.time() - last_step > 1:
                last_step = time.time()
                trials.trial(run_name, worlds, agents, kinds[i % len(kinds)])
                i += 1

def mohex_arena(run_name, worldfunc, agentfunc, device='cuda:1'):
    run_name = paths.resolve(run_name)
    # with logging.via_dir(run_name), stats.to_dir(run_name):
    trialer = mohex.Trialer(worldfunc, device)
    
    i = 0
    agent = None
    last_load, last_step = 0, 0
    while True:
        if time.time() - last_load > 15:
            last_load = time.time()
            agents = latest_agent(run_name, agentfunc, device=device)
            agent = list(agents.values())[0]
        
        if agent and (time.time() - last_step > 1):
            last_step = time.time()
            trialer.trial(agent)
            i += 1

@wraps(arena)
@contextmanager
def monitor(*args, **kwargs):
    set_start_method('spawn', True)
    p = Process(target=mohex_arena, args=args, kwargs=kwargs, name='arena-monitor')
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

def demo():
    from boardlaw import worldfunc, agentfunc
    from rebar import paths
    paths.clear('test')
    arena('test', worldfunc, agentfunc, ref_runs=['2020-11-27 19-40-27 az-test'])

def fill_matchups(run_name=-1, device='cuda'):
    from boardlaw import worldfunc, agentfunc
    from boardlaw.arena import matchups, periodic_agents, database, log

    run_name = paths.resolve(run_name)
    agents = periodic_agents(run_name, agentfunc, device=device)
    worlds = worldfunc(device=device, n_envs=256)

    n, w = database.symmetric_pandas(run_name, agents)

    while True:
        n, w = database.symmetric_pandas(run_name, agents)
        matchup = (n
            .stack()
            .loc[lambda s: s == 0]
            .reset_index()
            .loc[lambda df: df.black_name != df.white_name]
            .sample(1)
            .iloc[0, :2]
            .tolist())

        log.info(f'Playing {matchup}')
        matchup = {m: agents[m] for m in matchup}
        results = matchups.evaluate(worlds, matchup)

        wins, games = int(results[0].wins[0] + results[1].wins[1]), int(sum(r.games for r in results))
        log.info(f'Storing. {wins} wins in {games} games for {list(matchup)[0]} ')
        database.store(run_name, results)