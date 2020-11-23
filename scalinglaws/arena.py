import numpy as np
import pandas as pd
import pickle
import torch
from rebar import paths, storing, arrdict
from logging import getLogger
from . import analysis
from itertools import combinations, permutations

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

def league(worldfunc, agents, n_copies=32):
    worlds = worldfunc(n_envs=n_copies)
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

def parallel_league(worldfunc, agents, n_copies=32):
    n_agents = len(agents)

    idxs = np.arange(n_copies*n_agents*n_agents)
    fstmask, sndmask, _ = np.unravel_index(idxs, (n_agents, n_agents, n_copies))

    worlds = worldfunc(len(idxs))
    fstmask = torch.as_tensor(fstmask, device=worlds.device) 
    sndmask = torch.as_tensor(sndmask, device=worlds.device)

    terminations = torch.zeros((worlds.n_envs), device=worlds.device)
    rewards = torch.zeros((worlds.n_envs, 2), device=worlds.device)
    while True:
        log.info(f'Stepped; {(terminations >= 1).float().mean():.0%} have terminated')
        for (i, first) in enumerate(agents):
            mask = (fstmask == i) & (terminations < 1)
            if mask.any():
                decisions = agents[first](worlds[mask])
                worlds[mask], transitions = worlds[mask].step(decisions.actions)
                rewards[mask & (terminations < 1), 0] += (transitions.rewards[terminations[mask] < 1, 0] == 1)
                terminations[mask] += transitions.terminal
        
        for (j, second) in enumerate(agents):
            mask = (sndmask == j) & (terminations < 1)
            if mask.any():
                decisions = agents[second](worlds[mask])
                worlds[mask], transitions = worlds[mask].step(decisions.actions)
                rewards[mask & (terminations < 1), 1] += (transitions.rewards[terminations[mask] < 1, 1] == 1)
                terminations[mask] += transitions.terminal
        
        if (terminations >= 1).all():
            break


def run(worldfunc, agentfunc, run_name):
    agents = periodic_agents(agentfunc, run_name)
    agents['latest'] = latest_agent(agentfunc, run_name)


