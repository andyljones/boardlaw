import numpy as np
import pandas as pd
import pickle
import torch
from rebar import paths, storing, arrdict, numpy
from logging import getLogger
from IPython.display import clear_output

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

def summarize(vals, idxs, n_agents):
    if vals.ndim == 1:
        return summarize(vals[:, None], idxs, n_agents)[..., 0]

    D = vals.size(-1)
    totals = torch.zeros((n_agents*n_agents, D), device=vals.device)
    for d in range(D):
        totals[..., d].scatter_add_(0, idxs[:, 0]*n_agents + idxs[:, 1], vals[..., d].float())
    totals = totals.reshape((n_agents, n_agents, D))    
    return totals

def playoff(worldfunc, agents, n_copies=1):
    n_agents = len(agents)

    idxs = np.arange(n_copies*n_agents*n_agents)
    fstidxs, sndidxs, _ = np.unravel_index(idxs, (n_agents, n_agents, n_copies))

    worlds = worldfunc(len(idxs))
    idxs = torch.as_tensor(np.stack([fstidxs, sndidxs], -1), device=worlds.device) 

    while True:
        for seat in range(2):
            transitions = arrdict.arrdict(
                terminal=torch.zeros((worlds.n_envs), dtype=torch.bool, device=worlds.device),
                rewards=torch.zeros((worlds.n_envs, 2), device=worlds.device))
            for (i, first) in enumerate(agents):
                mask = (idxs[:, seat] == i) & (worlds.seats == seat)
                if mask.any():
                    decisions = agents[first](worlds[mask])
                    worlds[mask], masked_transitions = worlds[mask].step(decisions.actions)
                    transitions[mask] = masked_transitions

            yield transitions.map(summarize, idxs=idxs, n_agents=n_agents)

def accumulate(run_name, worldfunc, agents, **kwargs):
    writer = numpy.FileWriter(run_name)

    n_agents = len(agents)
    totals = arrdict.arrdict(
        terminal=np.zeros((n_agents, n_agents)),
        rewards=np.zeros((n_agents, n_agents, 2)))
    for step, summary in enumerate(playoff(worldfunc, agents, **kwargs)):
        summary = summary.cpu().numpy()
        totals += summary
        winrates = (2*totals.rewards[..., 0] - totals.terminal)/totals.terminal

        clear_output(wait=True)
        print(f'Step #{step}')
        print(f'Winrates:\n\n{winrates}')

        if any((summary > 0).any().values()):
            df = pd.concat({
                'rewards': pd.DataFrame(summary.rewards, agents.keys(), agents.keys()),
                'terminal': pd.DataFrame(summary.terminal, agents.keys(), agents.keys()),}, 1)
            record = {'-'.join(k): v for k, v in df.unstack().to_dict().items()}
            writer.write(record)

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