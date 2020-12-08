import sqlite3
import pandas as pd
from contextlib import contextmanager
from rebar import paths

DATABASE = 'output/arena.sql'

@contextmanager
def database():
    with sqlite3.connect(DATABASE) as conn:
        results_table = '''
            create table if not exists results(
                run_name text, 
                black_name text, white_name text, 
                black_wins real, white_wins real,
                PRIMARY KEY (run_name, black_name, white_name))'''
        conn.execute(results_table)
        yield conn

def store(run_name, result):
    if isinstance(result, list):
        for r in result:
            store(run_name, r)
        return 
    # upsert: https://stackoverflow.com/questions/2717590/sqlite-insert-on-duplicate-key-update-upsert
    with database() as conn:
        subs = (run_name, *result.names, *result.wins, *result.wins)
        conn.execute('''
            insert into results 
            values (?,?,?,?,?)
            on conflict(run_name, black_name, white_name) do update set 
            black_wins = black_wins + ?,
            white_wins = white_wins + ?''', subs)

def stored(run_name=''):
    run_name = paths.resolve(run_name)
    with database() as c:
        return pd.read_sql_query('select * from results where run_name like ?', c, params=(f'{run_name}%',))
    
def delete(run_name):
    with database() as c:
        c.execute('delete from results where run_name=?', (run_name,))

def summary(run_name):
    raw = stored(run_name)
    if len(raw) == 0:
        columns = pd.MultiIndex.from_product([['black_wins', 'white_wins',], []])
        return pd.DataFrame(columns=columns)
    df = (raw
            .groupby(['black_name', 'white_name'])
            [['black_wins', 'white_wins']]
            .sum()
            .unstack())
    
    names = sorted(list(set(df.index) | set(df.columns.get_level_values(1))))
    df = df.reindex(index=names).reindex(columns=names, level=1)
    return df.fillna(0)

def games(run_name):
    df = summary(run_name)
    if len(df) == 0:
        return pd.DataFrame()
    return df.white_wins + df.black_wins

def wins(run_name, min_games=-1):
    df = summary(run_name)
    if len(df) == 0:
        return pd.DataFrame()
    games = df.white_wins + df.black_wins
    games = games.where(games > min_games)
    return df.black_wins/games

def symmetric_games(run_name):
    g = games(run_name)
    return g + g.T

def symmetric_wins(run_name, min_games=-1):
    games = symmetric_games(run_name)
    df = summary(run_name)
    if len(df) == 0:
        return pd.DataFrame()
    return (df.black_wins + df.white_wins.T).where(games > min_games)