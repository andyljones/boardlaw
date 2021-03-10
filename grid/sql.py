# Right, time for a Serious Database for all this crap you've picked up
# * `run` table, with the run-specific stuff
# * `snapshot` table, with snapshot-specific stuff
# * `match` table, with results of matchups between tables. extra columns for extra arguments.
#
# Pandas SQL: https://pandas.pydata.org/docs/user_guide/io.html#io-sql-method

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey

Base = declarative_base()
class Run(Base):
    __tablename__ = 'runs'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    boardsize = Column(Integer)
    width = Column(Integer)
    depth = Column(Integer)
    nodes = Column(Integer)

class Snap(Base):
    __tablename__ = 'snap'

    id = Column(Integer, primary_key=True)
    run = Column(String, ForeignKey('run.id'))
    idx = Column(Integer)

class Agent(Base):
    __tablename__ = 'agent'

    id = Column(Integer, primary_key=True)
    snap = Column(Integer, ForeignKey('snap.id'))
    nodes = Column(Integer)

class Trial(Base):
    __tablename__ = 'trial'

    id = Column(Integer, primary_key=True)
    black_agent = Column(Integer, ForeignKey('agent.id'))
    white_agent = Column(Integer, ForeignKey('agent.id'))
    black_wins = Column(Integer)
    white_wins = Column(Integer)