import pandas as pd
import torch
from rebar import dotdict
from . import trials, mohex
from pavlov import storage, runs, logs, stats
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
    if 'agent' not in sd:
        # rebar legacy format
        sd = storage.expand(sd, 1)
    agent.load_state_dict(sd['agent'])
    return agent

def snapshot_agents(run_name, agentfunc, device='cpu', **kwargs):
    if not isinstance(run_name, (int, str)):
        agents = {}
        for r in run_name:
            agents.update(snapshot_agents(r, agentfunc, device=device, **kwargs))
        return agents

    try:
        stored = storage.snapshots(run_name)
    except ValueError:
        return {}
    else:
        agents = {} 
        for idx, info in stored.items():
            name = pd.Timestamp(info['_created']).strftime(r'%y%m%d-%H%M%S-snapshot')
            sd = storage.load(info['path'], device)
            agents[name] = assemble_agent(agentfunc, sd, device=device, **kwargs)
        return agents

def latest_agent(run_name, agentfunc, device='cpu', **kwargs):
    try:
        sd = storage.load_latest(run_name, device=device)
        return {'latest': assemble_agent(agentfunc, sd, device=device, **kwargs)}
    except ValueError:
        return {}

def snapshot_arena(run, worldfunc, agentfunc, device='cuda:1'):
    run = runs.resolve(run)
    with logs.to_run(run), stats.to_run(run):
        worlds = worldfunc(device=device, n_envs=256)

        i = 0
        agents = {}
        last_load, last_step = 0, 0
        while True:
            if time.time() - last_load > 15:
                last_load = time.time()
                agents = snapshot_agents(run, agentfunc, device=device)
            
            if time.time() - last_step > 1:
                last_step = time.time()
                trials.snapshot_trial(run, worlds, agents)
                i += 1

def mohex_arena(run, worldfunc, agentfunc, device='cuda:1'):
    run = runs.resolve(run)
    with logs.to_run(runs), stats.to_run(runs):
        trialer = mohex.Trialer(worldfunc, device)
        
        i = 0
        agent = None
        last_load, last_step = 0, 0
        while True:
            if time.time() - last_load > 15:
                last_load = time.time()
                agents = latest_agent(runs, agentfunc, device=device)
                if agents:
                    agent = list(agents.values())[0]
            
            if agent and (time.time() - last_step > 1):
                last_step = time.time()
                trialer.trial(agent)
                i += 1

@wraps(snapshot_arena)
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

def fill_matchups(run=-1, device='cuda:1', count=1):
    from boardlaw.main import worldfunc, agentfunc
    from boardlaw.arena import evaluator, snapshot_agents, database, log

    run = runs.resolve(run)
    agents = snapshot_agents(run, agentfunc, device=device)
    worlds = worldfunc(device=device, n_envs=256)

    while True:
        n, w = database.symmetric_pandas(run, agents)
        zeros = (n
            .stack()
            .loc[lambda s: s < count]
            .reset_index()
            .loc[lambda df: df.black_name != df.white_name])

        indices = {n: i for i, n in enumerate(n.index)}
        diff = abs(zeros.black_name.replace(indices) - zeros.white_name.replace(indices))
        ordered = zeros.loc[diff.sort_values().index]
        # Sample so there's no problems if we run in parallel
        matchup = ordered.head(10).sample(1).iloc[0, :2].tolist()

        log.info(f'Playing {matchup}')
        matchup = {m: agents[m] for m in matchup}
        results = evaluator.evaluate(worlds, matchup)

        wins, games = int(results[0].wins[0] + results[1].wins[1]), int(sum(r.games for r in results))
        log.info(f'Storing. {wins} wins in {games} games for {list(matchup)[0]} ')
        database.store(run, results)