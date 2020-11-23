import pandas as pd
import pickle
import torch
from rebar import paths, storing, arrdict
from logging import getLogger
from . import analysis
from itertools import permutations

log = getLogger(__name__)

def assemble_agent(agentfunc, sd):
    agent = agentfunc()
    agent.load_state_dict(sd['agent'])
    return agent

def periodic_agents(agentfunc, run_name):
    stored = storing.stored_periodic(run_name)
    challengers = {} 
    for _, row in stored.iterrows():
        name = row.date.strftime('%a-%H%M%S')
        sd = pickle.load(row.path.open('rb'))
        challengers[name] = assemble_agent(agentfunc, sd)
    return challengers

def latest_agent(agentfunc, run_name):
    sd = storing.load_latest(run_name)
    return assemble_agent(agentfunc, sd)

def run(worldfunc, agentfunc, run_name):
    agents = periodic_agents(agentfunc, run_name)
    agents['latest'] = latest_agent(agentfunc, run_name)

    worlds = worldfunc(n_envs=32)
    scores = arrdict.arrdict()
    for first, second in permutations(agents, 2):
        log.info(f'Evaluating {first} v {second}')
        trace = analysis.rollout(worlds, [agents[first], agents[second]], n_reps=1)

        # Mask out the first run from each environment. 
        # We're doing this to avoid biasing towards short runs.
        t = trace.transitions
        mask = (t.terminal.cumsum(0) <= 1).float()
        rewards = (t.rewards[..., 0] == 1)[mask].sum()
        terminals = t.terminal[mask].sum()
        scores[first, second] = (rewards/terminals)[0]

    return pd.Series(scores).apply(float).unstack()


