from contextlib import contextmanager
from matplotlib.pyplot import connect
import numpy as np
import pandas as pd
import pickle
import torch
from rebar import paths, storing, arrdict, numpy
from logging import getLogger
from IPython.display import clear_output
import sqlite3

log = getLogger(__name__)

DATABASE = 'output/arena.sql'

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

class Conductor:

    def __init__(self, worldfunc, agents, device='cpu', n_copies=1):
        self.worldfunc = worldfunc
        self.device = device
        self.agents = {k: agent.to(device) for k, agent in agents.items()}

        self.n_agents = len(self.agents)
        self.n_envs = n_copies*self.n_agents**2
        self.n_copies = n_copies

        self.worlds = None
        self.idxs = None
        self.seat = 0

        self.rewards = None

        self.initialize()

    def initialize(self):
        idxs = np.arange(self.n_envs)
        fstidxs, sndidxs, _ = np.unravel_index(idxs, (self.n_agents, self.n_agents, self.n_copies))

        self.worlds = self.worldfunc(len(idxs), self.device)
        self.idxs = torch.as_tensor(np.stack([fstidxs, sndidxs], -1), device=self.device) 

        self.rewards = torch.zeros((self.n_envs, self.worlds.n_seats))

    def step(self):
        terminal=torch.zeros((self.n_envs), dtype=torch.bool, device=self.device)
        for (i, name) in enumerate(self.agents):
            mask = (self.idxs[:, self.seat] == i) & (self.worlds.seats == self.seat)
            if mask.any():
                decisions = self.agents[name](self.worlds[mask])
                self.worlds[mask], transitions = self.worlds[mask].step(decisions.actions)
                terminal[mask] = transitions.terminal
                self.rewards[mask] += transitions.rewards

        self.seat = (self.seat + 1) % self.worlds.n_seats
        
        idxs = arrdict.numpyify(self.idxs)
        names = np.array(list(self.agents.keys()))[idxs[terminal]]
        rewards = arrdict.numpyify(self.rewards[terminal])

        self.rewards[terminal] = 0.

        return [(tuple(n), tuple(r)) for n, r in zip(names, rewards)]

@contextmanager
def database():
    with sqlite3.connect(DATABASE) as conn:
        results_table = '''
            create table if not exists results(id integer primary key, 
                run_name text, time text, 
                black_name text, white_name text, 
                black_reward real, white_reward real)'''
        conn.execute(results_table)
        yield conn

def store(run_name, results):
    timestamp = pd.Timestamp.now('utc').strftime('%Y-%m-%d %H:%M:%S.%fZ')
    with database() as conn:
        results = [(run_name, timestamp, *names, *map(float, rewards)) for names, rewards in results]
        conn.executemany('insert into results values (null,?,?,?,?,?,?)', results)

def read():
    with database() as c:
        return pd.read_sql_query('select * from results', c)

def summarize(vals, idxs, n_agents):
    if vals.ndim == 1:
        return summarize(vals[:, None], idxs, n_agents)[..., 0]

    D = vals.size(-1)
    totals = torch.zeros((n_agents*n_agents, D), device=vals.device)
    for d in range(D):
        totals[..., d].scatter_add_(0, idxs[:, 0]*n_agents + idxs[:, 1], vals[..., d].float())
    totals = totals.reshape((n_agents, n_agents, D))    
    return totals

def accumulate(run_name, worldfunc, agents, **kwargs):
    conductor = Conductor(worldfunc, agents, **kwargs)
    writer = numpy.FileWriter(run_name)

    count = 0
    n_agents = len(agents)
    totals = arrdict.arrdict(
        terminal=np.zeros((n_agents, n_agents)),
        rewards=np.zeros((n_agents, n_agents, 2)))
    while True:
        transitions, idxs = conductor.step()
        summary = transitions.map(summarize, idxs=idxs, n_agents=n_agents)

        totals += summary
        winrates = (totals.rewards[..., 0] + totals.terminal)/(2*totals.terminal)

        clear_output(wait=True)
        print(f'Step #{count}')
        print(f'Winrates:\n\n{winrates}\n')
        print(f'Terminals:\n\n{totals.terminal}')

        if any((summary > 0).any().values()):
            df = pd.concat({
                'rewards': pd.DataFrame(summary.rewards[..., 0], agents.keys(), agents.keys()),
                'terminal': pd.DataFrame(summary.terminal, agents.keys(), agents.keys()),}, 1)
            record = {'-'.join(k): v for k, v in df.unstack().to_dict().items()}
            writer.write(record)
        
        count += 1

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

    agents = {str(i): mohex.MoHexAgent(max_games=i) for i in [1, 10, 100, 1000, 10000]}

    def worldfunc(n_envs, device='cpu'):
        return hex.Hex.initial(n_envs=n_envs, boardsize=11, device=device)

    accumulate('output/mohex-tmp.npr', worldfunc, agents)