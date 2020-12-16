import pandas as pd
import numpy as np
import torch
from rebar import storing, logging, dotdict, stats, paths
import pickle
from . import database, matchups
from .. import mohex
import activelo
import time
from logging import getLogger
from contextlib import contextmanager
from functools import wraps
from multiprocessing import Process, set_start_method

# Re-export
from .plot import heatmap, periodic

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

def difference(soln, names, first, second):
    μ, Σ = pd.Series(soln.μ, names), pd.DataFrame(soln.Σ, names, names) 
    μd = μ[first] - μ[second]
    σ2d = Σ.loc[first, first] + Σ.loc[second, second] - 2*Σ.loc[first, second]
    return μd, σ2d**.5

def step_periodic(run_name, worlds, agents):
    n, w = database.symmetric_pandas(run_name, agents)
    log.info(f'Loaded {int(n.sum().sum())} games')

    valid = n.index[n.index.str.endswith('periodic')]
    if len(valid) < 2:
        raise ValueError('Need at least two periodic agents for a periodic step')
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
    if len(valid) < 2:
        raise ValueError('Need at least two periodic agents for a periodic step')
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

    wins = int(results[0].wins[0] + results[1].wins[1])
    games = int(sum(r.games for r in results))
    moves = int(sum(r.moves for r in results))
    log.info(f'Storing. {wins} wins in {games} games for {list(agents)[0]}; {moves/(2*games):.1f} average moves per player')
    stats.mean('moves-mohex', moves, 2*games)
    database.store(run_name, results)

_latest_matchup = None
_latest_results = []
def step_latest(run_name, worlds, agents):
    n, w = database.symmetric_pandas(run_name, agents)
    log.info(f'Loaded {int(n.sum().sum())} games')

    [latest] = n.index[n.index.str.endswith('latest')]
    first_periodic = n.index[n.index.str.endswith('periodic')][0]
    latest_periodic = n.index[n.index.str.endswith('periodic')][-1]
    matchup = (latest, latest_periodic)

    agents = {m: agents[m] for m in matchup}
    log.info('Playing ' + ' v. '.join(agents))
    results = matchups.evaluate(worlds.periodic, agents)

    global _latest_matchup, _latest_results
    if _latest_matchup != matchup:
        _latest_matchup = matchup
        _latest_results = []
    _latest_results.extend(results)

    log.info(f'Got {len(_latest_results)} results to accumulate')
    for r in _latest_results:
        w.loc[r.names[0], r.names[1]] += r.wins[0]
        w.loc[r.names[1], r.names[0]] += r.wins[1]
        n.loc[r.names[0], r.names[1]] += r.games
        n.loc[r.names[1], r.names[0]] += r.games
    
    wins, games = int(w.loc[matchup[0], matchup[1]]), int(n.loc[matchup[0], matchup[1]])
    log.info(f'Fitting posterior. {wins} wins for {list(agents)[0]} in {games} games')
    soln = activelo.solve(n.values, w.values)
    log.info(f'Fitted posterior, {(soln.σd**2).mean()**.5:.2f}σd over {n.shape[0]} agents')
    μ = pd.Series(soln.μ, n.index)

    μm, σm = difference(soln, n.index, latest, 'mohex')
    stats.mean_std('elo-mohex/latest', μm, σm)
    μ0, σ0 = difference(soln, n.index, latest, first_periodic)
    stats.mean_std('elo-first/latest', μ0, σ0)
    log.info(f'eElo for {latest} is {μ0:.2f}±{2*σ0:.2f} v. the first agent, {μm:.2f}±{2*σm:.2f} v. mohex')

    μm, σm = difference(soln, n.index, latest_periodic, 'mohex')
    stats.mean_std('elo-mohex/periodic', μm, σm)
    if latest_periodic != first_periodic:
        μ0, σ0 = difference(soln, n.index, latest_periodic, first_periodic)
        stats.mean_std('elo-first/periodic', μ0, σ0)
        log.info(f'eElo for {latest_periodic} is {μ0:.2f}±{2*σ0:.2f} v. the first agent, {μm:.2f}±{2*σm:.2f} v. mohex')
    else:
        log.info(f'eElo for {latest_periodic} is {μm:.2f}±{2*σm:.2f} v. mohex')


def step(run_name, worlds, agents, kind):
    log.info(f'Running a "{kind}" step')
    try:
        globals()[f'step_{kind}'](run_name, worlds, agents)
    except Exception as e:
        log.error(f'Failed while running a "{kind}" step with a "{e}" error')

def arena(run_name, worldfunc, agentfunc, device='cuda:1'):
    run_name = paths.resolve(run_name)
    with logging.to_dir(run_name), stats.to_dir(run_name):
        worlds = dotdict.dotdict(
            periodic=worldfunc(device=device, n_envs=256),
            latest=worldfunc(device=device, n_envs=256),
            mohex=worldfunc(device=device, n_envs=8))

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

def demo():
    from scalinglaws import worldfunc, agentfunc
    from rebar import paths
    paths.clear('test')
    arena('test', worldfunc, agentfunc, ref_runs=['2020-11-27 19-40-27 az-test'])

def fill_matchups(run_name=-1, device='cuda'):
    from scalinglaws import worldfunc, agentfunc
    from scalinglaws.arena import matchups, periodic_agents, database, log

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