import pandas as pd
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, ForeignKey, create_engine
from pavlov import runs, storage
import ast
from tqdm.auto import tqdm
from . import asymdata
from pathlib import Path
from contextlib import contextmanager

# First modern run
FIRST_RUN = pd.Timestamp('2021-02-03 12:47:26.557749+00:00')

DATABASE = Path('output/experiments/eval/database.sql')

Base = declarative_base()
class Run(Base):
    __tablename__ = 'runs'

    run = Column(String, primary_key=True)
    description = Column(String)
    boardsize = Column(Integer)
    width = Column(Integer)
    depth = Column(Integer)
    nodes = Column(Integer)

class Snap(Base):
    __tablename__ = 'snaps'

    id = Column(Integer, primary_key=True)
    run = Column(String, ForeignKey('runs.run'))
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
    insert = pd.concat([r.index.to_series().to_frame('run'), r[['description']], params[['boardsize', 'width', 'depth', 'nodes']]], 1)
    insert['nodes'] = insert.nodes.fillna(64)
    return insert.reset_index(drop=True)

def snapshot_data(r):
    snapshots = {}
    for _, r in tqdm(list(r.iterrows()), desc='snapshots'):
        for i, s in storage.snapshots(r.run).items():
            stored = storage.load_snapshot(r.run, i)
            if 'n_samples' in stored:
                snapshots[r.run, i] = {
                    'samples': stored['n_samples'], 
                    'flops': stored['n_flops']}
    snapshots = (pd.DataFrame.from_dict(snapshots, orient='index')
                    .rename_axis(index=('run', 'idx'))
                    .reset_index())
    return snapshots

def trial_agent_data(s):
    trials = pd.concat([asymdata.pandas(b) for b in range(3, 10)]).reset_index()

    regex = r'(?P<run>[\w-]+)\.(?P<idx>\d+)(?:\.(?P<nodes>\d+))?'
    black_agents = trials.black_name.str.extract(regex).fillna('64')
    white_agents = trials.white_name.str.extract(regex).fillna('64')

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

_engine = None
@contextmanager
def connection():
    if not DATABASE.parent.exists():
        DATABASE.parent.mkdir(exist_ok=True, parents=True)

    global _engine
    _engine = create_engine('sqlite:///' + str(DATABASE)) if _engine is None else _engine
    with _engine.connect() as conn:
        yield conn

def create():
    with connection() as conn:
        Base.metadata.create_all(conn.engine)

        r = run_data()
        r.to_sql('runs', conn, index=False, if_exists='append')

        s = snapshot_data(r)
        s.to_sql('snaps', conn, index=False, if_exists='append')

        a, t = trial_agent_data(s)
        a.to_sql('agents', conn, index=False, if_exists='append')
        t.to_sql('trials', conn, index=False, if_exists='append')

        conn.execute('''
            create view agents_details as
            select 
                agents.id, agents.nodes as test_nodes, 
                snaps.samples, snaps.flops as train_flops, snaps.idx, 
                runs.run, runs.description, runs.boardsize, runs.width, runs.depth, runs.nodes as train_nodes
            from agents
                inner join snaps on (agents.snap == snaps.id)
                inner join runs on (snaps.run == runs.run)''')

def run(sql):
    with connection() as conn:
        return conn.execute(sql)


def query(sql, **kwargs):
    with connection() as conn:
        return pd.read_sql_query(sql, conn, **kwargs)

def agent_query():
    return query('''select * from agents_details''', index_col='id')

def trial_query(boardsize, desc='%'):
    return query('''
        select trials.* 
        from trials 
            inner join agents_details as black
                on (trials.black_agent == black.id)
            inner join agents_details as white
                on (trials.white_agent == white.id)
        where (black.boardsize == ?) and (white.boardsize == ?) and (black.description like ?)''', index_col='id', params=(boardsize, boardsize, desc))