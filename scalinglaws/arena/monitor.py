import torch
from rebar import storing, logging
import pickle
from . import database, emcee
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

def periodic_agents(run_name, agentfunc, device='cpu'):
    if not isinstance(run_name, (int, str)):
        agents = {}
        for r in run_name:
            agents.update(periodic_agents(r, agentfunc, device))
        return agents

    try:
        stored = storing.stored_periodic(run_name)
    except ValueError:
        return {}
    else:
        agents = {} 
        for _, row in stored.iterrows():
            name = row.date.strftime(r'%y%m%d-%H%M%S')
            sd = pickle.load(row.path.open('rb'))
            agents[name] = assemble_agent(agentfunc, sd, device)
        return agents

def latest_agent(run_name, agentfunc, **kwargs):
    sd = storing.load_latest(run_name)
    return assemble_agent(agentfunc, sd, **kwargs)

def database_results(run_name, agents):
    agents = list(agents)
    wins = database.symmetric_wins(run_name).reindex(index=agents, columns=agents).fillna(0)
    games = database.symmetric_games(run_name).reindex(index=agents, columns=agents).fillna(0)
    return games.values, wins.values

def run(run_name, worldfunc, agentfunc, device='cpu', ref_runs=[], canceller=None, **kwargs):
    with logging.via_dir(run_name):
        matcher = emcee.Emcee(worldfunc, device=device, **kwargs)
        runs = ref_runs + [run_name]
        
        last_load, last_step = 0, 0
        while True:
            if time.time() - last_load > 60:
                last_load = time.time()
                agents = periodic_agents(runs, agentfunc, device)
                matcher.add_agents(agents)
            
            if time.time() - last_step > 1:
                last_step = time.time()
                games, wins = database_results(run_name, matcher.names.values())
                log.info(f'Loaded {games.sum()} games')
                soln = activelo.Solver(len(matcher.agents))(torch.as_tensor(games), torch.as_tensor(wins))
                log.info(f'Fitted a posterior, {(soln.σd**2).mean()**.5}σd')
                matchup = activelo.suggest(soln, matcher.n_envs)
                log.info(f'Suggestion is {matchup}')
                results = matcher.step(matchup)
                database.store(run_name, results)
                log.info(f'Stepped, stored')

            # #TODO: Hangs occasionally, and damned if I know why.
            # if canceller and canceller.wait(.1):
            #     log.info('Breaking')
            #     break

@wraps(run)
@contextmanager
def monitor(*args, **kwargs):
    canceller = Event()
    kwargs = {**kwargs, 'canceller': canceller}
    set_start_method('spawn', True)
    p = Process(target=run, args=args, kwargs=kwargs, name='arena-monitor')
    try:
        p.start()
        yield
    finally:
        log.info('Setting canceller')
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