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

def league_stats(worldfunc, agents, n_copies=1, n_reps=1):
    n_agents = len(agents)

    idxs = np.arange(n_copies*n_agents*n_agents)
    fstidxs, sndidxs, _ = np.unravel_index(idxs, (n_agents, n_agents, n_copies))

    worlds = worldfunc(len(idxs))
    fstidxs = torch.as_tensor(fstidxs, device=worlds.device) 
    sndidxs = torch.as_tensor(sndidxs, device=worlds.device)

    step = 0
    terminations = torch.zeros((worlds.n_envs), device=worlds.device)
    rewards = torch.zeros((worlds.n_envs, 2), device=worlds.device)
    while True:
        log.info(f'Step #{step}. {terminations.mean():.1f} average terminations; {(terminations >= n_reps).float().mean():.0%} worlds have hit {n_reps} terminations.')
        for (i, first) in enumerate(agents):
            mask = (fstidxs == i) & (worlds.seats == 0) & (terminations < n_reps)
            if mask.any():
                decisions = agents[first](worlds[mask])
                worlds[mask], transitions = worlds[mask].step(decisions.actions)
                rewards[mask & (terminations < n_reps)] += transitions.rewards[terminations[mask] < n_reps]
                terminations[mask] += transitions.terminal
        
        for (j, second) in enumerate(agents):
            mask = (sndidxs == j) & (worlds.seats == 1) & (terminations < n_reps)
            if mask.any():
                decisions = agents[second](worlds[mask])
                worlds[mask], transitions = worlds[mask].step(decisions.actions)
                rewards[mask & (terminations < n_reps)] += transitions.rewards[terminations[mask] < n_reps]
                terminations[mask] += transitions.terminal
        
        if (terminations >= n_reps).all():
            break

        step += 1

    totals = torch.zeros((n_agents*n_agents, 2), device=worlds.device)
    totals[..., 0].scatter_add_(0, fstidxs*n_agents + sndidxs, rewards[..., 0])
    totals[..., 1].scatter_add_(0, fstidxs*n_agents + sndidxs, rewards[..., 1])
    totals = totals.reshape((n_agents, n_agents, 2))    

    winrates = 1/2*(totals[..., 0]/(n_copies*n_reps)) + .5

    return pd.DataFrame(winrates.cpu().numpy(), agents.keys(), agents.keys())

def league(worldfunc, agents):
    pass

def plot_confusion(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    with plt.style.context('seaborn-poster'):
        ax = sns.heatmap(df, cmap='RdBu', annot=True, vmin=0, vmax=1, annot_kws={'fontsize': 'large'})
        ax.set_xlabel('white')
        ax.set_ylabel('black')

def stddev(df, n_trajs):
    alpha = df*n_trajs + 1
    beta = n_trajs + 1 - df*n_trajs
    return (alpha*beta/((alpha + beta)**2 * (alpha + beta + 1)))**.5 

def run(worldfunc, agentfunc, run_name):
    agents = periodic_agents(agentfunc, run_name)
    agents['latest'] = latest_agent(agentfunc, run_name)


def mohex_calibration():
    from . import mohex

    agents = {str(i): mohex.MoHexAgent(max_games=i) for i in [1, 10, 100, 1000]}

    def worldfunc(n_envs, device='cuda'):
        return hex.Hex.initial(n_envs=n_envs, boardsize=11, device=device)

    df = parallel_league(worldfunc, agents, n_reps=10)