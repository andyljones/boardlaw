import sqlite3
import pandas as pd
from contextlib import contextmanager

DATABASE = 'output/arena.sql'

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
        results = [(run_name, timestamp, *map(str, names), *map(float, rewards)) for names, rewards in results]
        conn.executemany('insert into results values (null,?,?,?,?,?,?)', results)

def stored(run_name=''):
    with database() as c:
        return pd.read_sql_query('select * from results where run_name like ?', c, params=(f'{run_name}%',))
    
def delete(run_name):
    with database() as c:
        c.execute('delete from results where run_name=?', (run_name,))
