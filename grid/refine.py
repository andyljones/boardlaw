import aljpy
from tqdm.auto import tqdm 
import activelo
from boardlaw.arena import analysis
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
from . import aws, vast
from pavlov import stats
from invoke.exceptions import UnexpectedExit

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

def evaluate(run, idx, max_games=128, target_std=.025):
    """
    Memory usage:
        * 3b1w2d: 1.9G
        * 9b4096w1d: 2.5G

    OK, '2021-02-08 23-10-31 safe-tool' takes 
        * 40s/game on a 2-CPU, 4GB Google Cloud machine. Seems to use both CPUs.
        * 30s/game on my local server
    """

    import os
    if 'JITTENS_GPU' in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['JITTENS_GPU']
        print(f'Devices set to "{os.environ["CUDA_VISIBLE_DEVICES"]}"')
        device = 'cuda'
    else:
        device = 'cpu'

    assure(run, idx)
    worlds = common.worlds(run, 2, device)
    agent = common.agent(run, idx, device)
    arena = mohex.CumulativeArena(worlds)

    # Warm things up; get everything compiled
    decisions = agent(worlds)
    worlds.step(decisions.actions)

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

def submit(query='width <= 64', resources={'cpu': 2, 'memory': 4}):
    if query is None:
        submit('width <= 64', {'cpu': 2, 'memory': 4})
        submit('width > 64', {'gpu': 1})

    df = runs.pandas().loc[lambda df: df.description.fillna("").str.startswith("main/")]
    df = pd.concat([df, pd.DataFrame(df.params.values.tolist(), df.index)], 1)
    df['n_snapshots'] = df._files.apply(lambda d: len([f for f in d if f.startswith('storage.snapshot')]))
    df = (df
        .reset_index()
        .query(query)
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
            resources=resources)
        
def fetch():
    for id, machine in jittens.machines.machines().items(): 
        if hasattr(machine, 'connection'):
            conn = machine.connection
            [keyfile] = conn.connect_kwargs['key_filename']
            ssh = f"ssh -o StrictHostKeyChecking=no -i '{keyfile}' -p {conn.port}"
            command = f"""rsync -Rr --port 12000 -e "{ssh}" {conn.user}@{conn.host}:"/code/./*/output/pavlov/*/*.json" "output/refine" """
        else:
            command = """rsync -Rr .jittens/local/./*/output/pavlov/*/*.json "output/refine" """
        
        try:
            invoke.context.Context().run(command, hide=True)
        except UnexpectedExit:
            log.exception(f'Exception fetching from {id}')
            pass

def refresh():
    vast.jittenate(local=True)
    while True:
        jittens.manage.refresh()
        time.sleep(5)
        fetch()


def arena_results():
    import json
    import pandas as pd
    from pathlib import Path

    df = []
    globs = (
        list(Path('output/refine').glob('*/output/pavlov/*')) +
        list(Path('output/refine-0').glob('*')) + 
        list(Path('output/refine-1').glob('*')))
    for p in globs:
        try:
            info = json.loads((p / '_info.json').read_text())
            arena = json.loads((p / 'arena.json').read_text())
            for a in arena:
                df.append({'run': p.name, **info['params'], **a})
            
        except FileNotFoundError:
            pass
    df = pd.DataFrame(df)

    return df

def keyed_arena():
    df = arena_results()

    keys = {}
    for i, r in df.iterrows():
        ks = {}
        for color in ['black', 'white']:
            name = r[f'{color}_name']
            key = f'{color}_key'
            if name.startswith('mohex'):
                ks[key] = f'{r.boardsize}-{name}'
            elif name == 'latest':
                ks[key] = f'{r.boardsize}B{r.width}W{r.depth}D+S'
            else:
                idx = name.split('.')[1] 
                ks[key] = f'{r.boardsize}B{r.width}W{r.depth}D{idx}S'
        keys[i] = ks
    df = pd.concat([df, pd.DataFrame.from_dict(keys, orient='index')], 1)

    return df


def keyed_samples():
    aux = (keyed_arena()
        .groupby('black_key').first()
        .loc[lambda df: ~df.index.str.contains('mohex')]
        .loc[:, ['run', 'boardsize', 'width', 'depth', 'black_name']])

    trans = {}
    for r, g in tqdm(aux.groupby('run')):
        times = {f'snapshot.{idx}': pd.to_datetime(info['_created']) for idx, info in storage.snapshots(r).items()}
        
        samples = pd.DataFrame(stats.array(r, 'count.samples')).set_index('_time').tz_localize('UTC').cumsum()['total']
        samps = {}
        for s, t in times.items():
            samps[s] = samples[:t].iloc[-1]
        # This is approximate and I hate it
        samps['latest'] = samples.iloc[-1]
        
        s = pd.Series(samps)[g.black_name.values]
        s.index = g.index
        trans[r] = s
    trans = pd.concat(trans).reset_index()
    trans.columns = ['run', 'black_key', 'samples']

    aux = pd.concat([aux, trans.set_index('black_key')['samples']], 1)

def solve(t, games, wins):
    b = t[0]

    refs = list(set(games.index[games.index.str.startswith(f'{b}-mohex')]))
    query = [t] + refs
    soln = activelo.solve(games.loc[query, query], wins.loc[query, query])

    μd, σd = analysis.difference(soln, f'{b}-mohex-0.00', t)
    return {'μ': μd, 'σ': σd}


def solutions():
    keyed = keyed_arena()[['black_key', 'white_key', 'black_wins', 'white_wins']]
    for b in [3, 5, 7, 9]:
        aux = database.pandas(f'mohex-{b}').reset_index()
        aux['black_key'] = f'{b}-' + aux.black_name
        aux['white_key'] = f'{b}-' + aux.white_name
        keyed = pd.concat([keyed, aux.reindex(columns=keyed.columns)])

    black_wins = pd.pivot_table(keyed, 'black_wins', 'black_key', 'white_key', aggfunc='sum').fillna(0)
    white_wins = pd.pivot_table(keyed, 'white_wins', 'black_key', 'white_key', aggfunc='sum').fillna(0)

    wins = black_wins + white_wins.T
    games = black_wins + white_wins + black_wins.T + white_wins.T

    targets = set(keyed.black_key.values) | set(keyed.white_key.values)
    targets = [t for t in targets if 'mohex' not in t]

    with aljpy.parallel(solve) as p:
        solns = p.wait({t: p(t, games, wins) for t in targets})

    solns = pd.DataFrame.from_dict(solns, orient='index')

    aux = keyed_samples()
    joint = pd.concat([aux, solns], 1)

    return joint