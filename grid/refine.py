import numpy as np
import pandas as pd
import invoke
import time
from boardlaw.arena import common, mohex, database
from logging import getLogger
from rebar import arrdict
from boardlaw import backup
from pavlov import runs, storage
from shlex import quote
import jittens
from . import aws

log = getLogger(__name__)

def rename(r, new):
    if isinstance(r, list):
        return [rename(rr, new) for rr in r]

    r = r.copy()
    r['names'] = [(new if n == 'agent' else n) for n in r['names']]
    return r

def assure(run, idx=None):
    if not runs.exists(run):
        p = runs.path(run, res=False)
        p.mkdir(exist_ok=True, parents=True)

        state_file = 'storage.latest.pkl' if idx is None else f'storage.snapshot.{idx}.pkl'
        for file in [state_file, 'storage.named.model.pkl', '_info.json']:
            backup.download(str(p / file), f'boardlaw:output/pavlov/{run}/{file}')

def evaluate(run, idx, max_games=32, target_std=.025):
    """
    Memory usage:
        * 3b1w2d: 1.9G
        * 9b4096w1d: 2.5G

    OK, '2021-02-08 23-10-31 safe-tool' takes 
        * 40s/game on a 2-CPU, 4GB Google Cloud machine. Seems to use both CPUs.
        * 30s/game on my local server
    """

    assure(run, idx)
    worlds = common.worlds(run, 2)
    agent = common.agent(run, idx)
    arena = mohex.CumulativeArena(worlds)

    name = 'latest' if idx is None else f'snapshot.{idx}'

    start = time.time()
    trace = []
    while True:
        soln, results = arena.play(agent)
        trace.append(soln)

        rate = (time.time() - start)/(soln.games + 1e-6)
        log.info(f'{rate:.0f}s per game; {rate*soln.games:.0f}s so far, {rate*max_games:.0f}s expected')

        database.save(run, rename(results, name))

        if soln.std < target_std:
            break
        if soln.games >= max_games:
            break

    return arrdict.stack(trace), results

def launch():
    df = runs.pandas().loc[lambda df: df.description.fillna("").str.startswith("main/")]
    df = pd.concat([df, pd.DataFrame(df.params.values.tolist(), df.index)], 1)
    df['n_snapshots'] = df._files.apply(lambda d: len([f for f in d if f.startswith('storage.snapshot')]))
    df = (df
        .reset_index()
        .groupby(['boardsize', 'width', 'depth'])
        .apply(lambda g: g[g.n_snapshots == g.n_snapshots.max()].iloc[-1])
        .set_index('run'))

    invocations = []
    for run in df.index:
        for snapshot in list(storage.snapshots(run)) + [None]:
            invocations.append((run, snapshot))
    invocations = np.random.permutation(invocations)
    
    archive = jittens.jobs.compress('.', '.jittens/bulk.tar.gz', ['credentials.json'])
    for run, snapshot in invocations:
        jittens.jobs.submit(
            cmd=f"""python -c "from grid.refine import *; evaluate({quote(run)}, {snapshot})" >logs.txt 2>&1""", 
            archive=archive, 
            resources={'cpu': 2, 'memory': 4})
        
    aws.jittenate()
    while True:
        jittens.manage.refresh()
        time.sleep(5)
        fetch()

def fetch():
    for id, machine in jittens.machines.machines().items(): 
        conn = machine.connection
        [keyfile] = conn.connect_kwargs['key_filename']
        ssh = f"ssh -o StrictHostKeyChecking=no -i '{keyfile}' -p {conn.port}"
        
        command = f"""rsync -Rr --port 12000 -e "{ssh}" {conn.user}@{conn.host}:"/code/*/output/pavlov/./*/*.json" "output/refine" """
        invoke.context.Context().run(command)

def observed_rates():
    import json
    import pandas as pd
    from pathlib import Path

    df = []
    for p in Path('output/refine').glob('*'):
        try:
            info = json.loads((p / '_info.json').read_text())
            arena = json.loads((p / 'arena.json').read_text())
            games = sum([a['black_wins'] + a['white_wins'] for a in arena])
            
            # start = pd.Timestamp(info['_files']['arena.json']['_created'])
            # duration = (ended - start).total_seconds()
            
            df.append({**info['params'], 'games': games})
        except FileNotFoundError:
            pass
    df = pd.DataFrame(df)

    return df