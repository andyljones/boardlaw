import pandas as pd
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, ForeignKey, create_engine
from pavlov import runs, storage
import ast
from tqdm.auto import tqdm
from . import asymdata

# First modern run
FIRST_RUN = pd.Timestamp('2021-02-03 12:47:26.557749+00:00')

DATABASE = 'output/experiments/eval/database.sql'

Base = declarative_base()
class Run(Base):
    __tablename__ = 'runs'

    name = Column(String, primary_key=True)
    description = Column(String)
    boardsize = Column(Integer)
    width = Column(Integer)
    depth = Column(Integer)
    nodes = Column(Integer)

class Snap(Base):
    __tablename__ = 'snaps'

    id = Column(Integer, primary_key=True)
    run = Column(String, ForeignKey('runs.name'))
    idx = Column(Integer)
    samples = Column(Float)
    flops = Column(Float)

class Agent(Base):
    __tablename__ = 'agents'

    id = Column(Integer, primary_key=True)
    snap = Column(Integer, ForeignKey('snaps.id'))
    nodes = Column(Integer)
    c = Column(Float)

class Trial(Base):
    __tablename__ = 'trials'

    id = Column(Integer, primary_key=True)
    black_agent = Column(Integer, ForeignKey('agents.id'))
    white_agent = Column(Integer, ForeignKey('agents.id'))
    black_wins = Column(Integer)
    white_wins = Column(Integer)
    moves = Column(Integer)
    times = Column(Integer)

def run_data():
    r = runs.pandas().loc[lambda df: df._created >= FIRST_RUN]
    params = r.params.dropna().apply(pd.Series).reindex(r.index)
    insert = pd.concat([r.index.to_series().to_frame('name'), params[['boardsize', 'width', 'depth', 'nodes']]], 1)
    insert['nodes'] = insert.nodes.fillna(64)
    return insert

def snapshot_data(r):
    snapshots = {}
    for r, _ in tqdm(list(r.iterrows()), desc='snapshots'):
        for i, s in storage.snapshots(r).items():
            stored = storage.load_snapshot(r, i)
            if 'n_samples' in stored:
                snapshots[r, i] = {
                    'samples': stored['n_samples'], 
                    'flops': stored['n_flops']}
    snapshots = (pd.DataFrame.from_dict(snapshots, orient='index')
                    .rename_axis(index=('run', 'idx'))
                    .reset_index())
    return snapshots

def trial_agent_data(s):
    trials = pd.concat([asymdata.pandas(b) for b in range(3, 10)]).reset_index()

    regex = r'(?P<run>[\w-]+)\.(?P<idx>\d+)(?:\.(?P<nodes>\d+))?'
    black_agents = trials.black_name.str.extract(regex).fillna(64)
    white_agents = trials.white_name.str.extract(regex).fillna(64)

    agents = (pd.concat([black_agents, white_agents])
                .drop_duplicates(['run', 'idx', 'nodes'])
                .reset_index(drop=True)
                .rename_axis(index='id')
                .reset_index('id'))

    short = pd.concat({
            'run': s.run.str.extract(r'.* (?P<nickname>[\w-]+)$', expand=False), 
            'idx': s.idx.astype(str), 
            'id': s.index.to_series()}, 1)
    agents = (pd.merge(agents, short, how='left', on=['run', 'idx'], suffixes=('', '_'))
                .rename(columns={'id_': 'snap'}))
    agents['c'] = 1/16

    trials['black_agent'] = pd.merge(black_agents, agents, on=['run', 'idx', 'nodes'], how='left')['id']
    trials['white_agent'] = pd.merge(white_agents, agents, on=['run', 'idx', 'nodes'], how='left')['id']
    trials = (trials
                .rename_axis(index='id')
                .reset_index()
                [['id', 'black_agent', 'white_agent', 'black_wins', 'white_wins', 'moves', 'times']])

    return agents[['id', 'snap', 'nodes', 'c']], trials


def create():
    engine = create_engine('sqlite:///' + DATABASE)
    with engine.connect() as conn:
        Base.metadata.create_all(engine)

        r = run_data()
        r.to_sql('runs', conn, if_exists='replace')

        s = snapshot_data(r)
        s.to_sql('snaps', conn, if_exists='replace')

        a, t = trial_agent_data(s)
        a.to_sql('agents', conn, if_exists='replace')
        t.to_sql('trials', conn, if_exists='replace')





