# Right, time for a Serious Database for all this crap you've picked up
# * `run` table, with the run-specific stuff
# * `snapshot` table, with snapshot-specific stuff
# * `match` table, with results of matchups between tables. extra columns for extra arguments.
#
# Pandas SQL: https://pandas.pydata.org/docs/user_guide/io.html#io-sql-method

import pandas as pd
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, ForeignKey, create_engine
from pavlov import runs

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

def create():
    engine = create_engine('sqlite:///:memory')
    with engine.connect() as conn:
        rs = runs.pandas().loc[lambda df: df._created >= FIRST_RUN]
        params = rs.params.dropna().apply(pd.Series).reindex(rs.index)
        insert = pd.concat([rs.index.to_series().to_frame('name'), params[['boardsize', 'width', 'depth', 'nodes']]], 1)
        insert['nodes'] = insert.nodes.fillna(64)
        insert.to_sql('runs', conn, if_exists='append')





