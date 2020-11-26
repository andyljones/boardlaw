from rebar import storing
import pickle
from . import database, matchups

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

def run(run_name, worldfunc, device='cpu'):
    matcher = matchups.Matcher(worldfunc, device=device)
    while True:
        pass