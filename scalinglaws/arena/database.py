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
    # upsert: https://stackoverflow.com/questions/2717590/sqlite-insert-on-duplicate-key-update-upsert
    with database() as conn:
        subs = (
            run_name, result.black_name, result.white_name, result.black_wins, result.white_wins, 
            result.black_wins, result.white_wins)
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
    return (stored(run_name)
            .groupby(['black_name', 'white_name'])
            [['black_wins', 'white_wins']]
            .sum()
            .unstack()
            .fillna(0))

def games(run_name):
    df = summary(run_name)
    return (df.white_wins + df.black_wins).sum()

def winrate(run_name, min_games=256):
    df = summary(run_name)
    games = df.white_wins + df.black_wins
    games = games.where(games > min_games)
    return df.black_wins/games

def symmetric_winrate(run_name, min_games=256):
    df = summary(run_name)
    games = df.black_wins + df.white_wins
    games = games.where(games > min_games)
    return .5*df.black_wins/games + .5*df.white_wins.T/games.T

def _transfer():
    old = stored()

    new = (old
        .assign(black_wins=lambda df: df.black_reward == 1, white_wins=lambda df: df.white_reward == 1)
        .groupby(['run_name', 'black_name', 'white_name'])[['black_wins', 'white_wins']].sum()
        .reset_index())

    from tqdm.auto import tqdm
    import aljpy

    for _, row in tqdm(new.iterrows(), total=len(new)):
        database.store(row.run_name, aljpy.dotdict(row))
        