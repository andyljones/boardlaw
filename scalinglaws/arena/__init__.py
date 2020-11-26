import numpy as np
import pandas as pd
import pickle
from rebar import storing, arrdict
from logging import getLogger
from IPython.display import clear_output
from . import matchups, database

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

def relative_elos(expected):
    # https://www.remi-coulom.fr/Bayesian-Elo/#theory
    return (400*(np.log10(expected) - np.log10(1 - expected))).round(0).astype(int)

def stddev(df, n_trajs):
    alpha = df*n_trajs + 1
    beta = n_trajs + 1 - df*n_trajs
    return (alpha*beta/((alpha + beta)**2 * (alpha + beta + 1)))**.5 

def mohex_calibration():
    from . import mohex

    agents = {str(i): mohex.MoHexAgent(max_games=i) for i in [1, 10, 100, 1000, 10000]}

    def worldfunc(n_envs, device='cpu'):
        return hex.Hex.initial(n_envs=n_envs, boardsize=11, device=device)

    matcher = matchups.Matcher(worldfunc, agents)

    step = 0
    while True:
        clear_output(wait=True)
        print('Step #{step}')
        step += 1

        results = matcher.step()
        database.store('mohex-manual', results)