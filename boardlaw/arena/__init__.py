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
from .plot import heatmap, snapshots, nontransitivities
from .analysis import elos

log = getLogger(__name__)

def assemble_agent(agentfunc, sd):
    agent = agentfunc()
    if 'agent' not in sd:
        # rebar legacy format
        sd = storage.expand(sd, 1)
    agent.load_state_dict(sd['agent'])
    return agent

def snapshot_agents(run, agentfunc, **kwargs):
    if not isinstance(run, (int, str)):
        agents = {}
        for r in run:
            agents.update(snapshot_agents(r, agentfunc, **kwargs))
        return agents

    period = kwargs.get('period', 1)
    tail = kwargs.get('tail', int(1e6))
    try:
        stored = pd.DataFrame.from_dict(storage.snapshots(run), orient='index').tail(tail).iloc[::period]
    except ValueError:
        return {}
    else:
        agents = {} 
        for idx, info in stored.iterrows():
            if idx % period == 0:
                name = pd.Timestamp(info['_created']).strftime(r'%y%m%d-%H%M%S-snapshot')
                sd = storage.load_path(info['path'])
                agents[name] = assemble_agent(agentfunc, sd)
        return agents

def latest_agent(run_name, agentfunc, device='cuda', **kwargs):
    try:
        sd = storage.load_latest(run_name, device=device)
        return {'latest': assemble_agent(agentfunc, sd, **kwargs)}
    except FileNotFoundError:
        return {}

def snapshot_arena(run, worldfunc, agentfunc, device='cuda'):
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

def mohex_arena(run, worldfunc, agentfunc):
    log.info('Arena launched')
    if isinstance(run, tuple):
        source_run = runs.resolve(run[0])
        stats_run = runs.resolve(run[1])
    else:
        source_run = runs.resolve(run)
        stats_run = runs.resolve(run)

    log.info(f'Running arena for "{source_run}", storing in "{stats_run}"')
    with logs.to_run(stats_run), stats.to_run(stats_run):
        trialer = mohex.Trialer(worldfunc)
        
        i = 0
        agent = None
        last_load, last_step = 0, 0
        while True:
            if time.time() - last_load > 15:
                last_load = time.time()
                agents = latest_agent(source_run, agentfunc)
                if agents:
                    agent = list(agents.values())[0]
                else:
                    log.info('No agents yet')
            
            if agent and (time.time() - last_step > 1):
                last_step = time.time()
                log.info('Running trial')
                trialer.trial(agent)
                i += 1

@wraps(snapshot_arena)
@contextmanager
def monitor(*args, **kwargs):
    set_start_method('spawn', True)
    p = Process(target=mohex_arena, args=args, kwargs=kwargs, name='arena-monitor')
    try:
        p.start()
        yield p
    finally:
        for _ in range(50):
            if not p.is_alive():
                log.info('Arena monitor dead')
                break
            time.sleep(.1)
        else:
            log.info('Abruptly terminating arena monitor; it should have shut down naturally!')
            p.terminate()

def matchups(run=-1, count=1, **kwargs):
    from boardlaw.hex import Hex
    from boardlaw import mcts
    from boardlaw.arena import evaluator, snapshot_agents, database, log

    run = runs.resolve(run)
    agentfunc = lambda: mcts.MCTSAgent(storage.load_raw(run, 'model'))
    agents = snapshot_agents(run, agentfunc, **kwargs)
    worlds = Hex.initial(256, boardsize=runs.info(run)['params']['boardsize'])

    while True:
        agents = snapshot_agents(run, agentfunc, **kwargs)

        n, w = database.symmetric(run, agents)
        zeros = (n
            .stack()
            .loc[lambda s: s < count]
            .reset_index()
            .loc[lambda df: df.black_name != df.white_name])

        indices = {n: i for i, n in enumerate(n.index)}
        diff = abs(zeros.black_name.replace(indices) - zeros.white_name.replace(indices))
        ordered = zeros.loc[diff.sort_values().index]
        # Sample so there's no problems if we run in parallel
        if len(ordered) == 0:
            log.info('No matchups to play')
            time.sleep(15)
            continue
        matchup = ordered.head(10).sample(1).iloc[0, :2].tolist()

        log.info(f'Playing {matchup}')
        matchup = {m: agents[m] for m in matchup}
        results = evaluator.evaluate(worlds, matchup)

        wins, games = int(results[0].wins[0] + results[1].wins[1]), int(sum(r.games for r in results))
        log.info(f'Storing. {wins} wins in {games} games for {list(matchup)[0]} ')
        database.save(run, results)