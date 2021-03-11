import pandas as pd
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, ForeignKey, create_engine
from pavlov import runs, storage
import ast
from tqdm.auto import tqdm

# First modern run
FIRST_RUN = pd.Timestamp('2021-02-03 12:47:26.557749+00:00')

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

    run = Column(String, ForeignKey('runs.id'), primary_key=True)
    idx = Column(Integer, primary_key=True)
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

def snaps(rs):
    snapshots = {}
    for r, _ in tqdm(list(rs.iterrows()), desc='snapshots'):
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

def create():
    engine = create_engine('sqlite:///:memory:')
    with engine.connect() as conn:
        rs = runs.pandas().loc[lambda df: df._created >= FIRST_RUN]
        params = rs.params.dropna().apply(pd.Series).reindex(rs.index)
        insert = pd.concat([rs.index.to_series().to_frame('name'), params[['boardsize', 'width', 'depth', 'nodes']]], 1)
        insert['nodes'] = insert.nodes.fillna(64)
        insert.to_sql('runs', conn, if_exists='replace')

        ss = snaps(rs)
        ss.to_sql('snaps', conn, if_exists='replace')




