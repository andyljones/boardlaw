import time
import numpy as np
import activelo
import pandas as pd
from boardlaw import arena
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method
from geotorch.exceptions import InManifoldError
from logging import getLogger
from . import data, asymdata
from rebar.parallel import CUDAPoolExecutor

log = getLogger(__name__)

set_start_method('spawn', True)

N_ENVS = 1024

def compile(name):
    run, idx = name.split('.')
    log.info('Compiling...')
    agent = arena.common.agent(f'*{run}', int(idx), 'cuda')
    worlds = arena.common.worlds(f'*{run}', 2, 'cuda')
    
    decisions = agent(worlds)
    worlds.step(decisions.actions)
    log.info('Compiled')

def evaluate(Aname, Bname):
    Arun, Aidx = Aname.split('.')
    Brun, Bidx = Bname.split('.')
    A = arena.common.agent(f'*{Arun}', int(Aidx), 'cuda')
    B = arena.common.agent(f'*{Brun}', int(Bidx), 'cuda')
    worlds = arena.common.worlds(f'*{Arun}', N_ENVS, 'cuda')

    return arena.common.evaluate(worlds, [(Aname, A), (Bname, B)]), worlds.boardsize

def update(games, wins, results):
    games, wins = games.copy(), wins.copy()
    for result in results:
        games.loc[result.names[0], result.names[1]] += result.games
        games.loc[result.names[1], result.names[0]] += result.games
        wins.loc[result.names[0], result.names[1]] += result.wins[0]
        wins.loc[result.names[1], result.names[0]] += result.wins[1]
    return games, wins

def solve(games, wins, soln=None):
    try:
        return activelo.solve(games, wins, soln=soln)
    except InManifoldError:
        log.warning('Got a manifold error; throwing soln out')
        return None

def activelo_suggest(soln):
    #TODO: Can I use the eigenvectors of the Σ to rapidly make orthogonal suggestions
    # for parallel exploration? Do I even need to go that complex - can I just collapse
    # Σ over the in-flight pairs?
    imp = activelo.improvement(soln)
    idx = np.random.choice(imp.stack().index, p=imp.values.flatten()/imp.sum().sum())
    return tuple(idx)

def activelo_eval(boardsize=9, n_workers=6):
    snaps = data.snapshot_solns(boardsize, solve=False)
    games, wins = data.load(boardsize, snaps.index)

    compile(snaps.index[0])

    solver, soln, σ = None, None, None
    futures = {}
    with CUDAPoolExecutor(n_workers+1) as pool:
        while True:
            if solver is None:
                log.info('Submitting solve task')
                solver = pool.submit(solve, games, wins)
            elif solver.done():
                soln = solver.result()
                solver = None
                if soln is not None:
                    μ, σ = arena.analysis.difference(soln, soln.μ.idxmin())
                    log.info(f'μ_max: {μ.max():.1f}')
                    log.info(f'σ_ms: {σ.pow(2).mean()**.5:.2f}')

            for key, future in list(futures.items()):
                if future.done():
                    results, boardsize = future.result()
                    games, wins = update(games, wins, results)
                    del futures[key]
                    data.save(boardsize, games, wins)
                    
                    log.info(f'saturation: {games.sum().sum()/N_ENVS/games.shape[0]:.0%}')

            while len(futures) < n_workers:
                if soln is None:
                    sugg = tuple(np.random.choice(games.index, (2,)))
                else:
                    sugg = activelo_suggest(soln)
                
                log.info('Submitting eval task')
                futures[(np.random.randint(2**32), *sugg)] = pool.submit(evaluate, *sugg)


class StructuredSuggester:

    def __init__(self, snaps):
        self.games, self.wins = {}, {}
        for b in snaps.boardsize.unique():
            self.games[b], self.wins[b] = data.load(b, snaps[snaps.boardsize == b].index)

        self.start = time.time()
        self.init_matches = self.played()
        self.moves = 0 
    
    def played(self):
        return sum(g.gt(0).sum().sum() for g in self.games.values())

    def update(self, results, boardsize):
        self.games[boardsize], self.wins[boardsize] = update(self.games[boardsize], self.wins[boardsize], results)

    def report(self, sugg):
        matches_played = self.played() - self.init_matches
        time_passed = (time.time() - self.start)
        match_rate = (matches_played+1)/time_passed

        matches_remain = len(sugg)
        time_remain = pd.to_timedelta(matches_remain/match_rate, unit='s')
        end_time = pd.Timestamp.now() + time_remain

        secs_remain = time_remain.total_seconds()
        log.info(f'{60*match_rate:.0f} matches/min, {secs_remain/3600:.0f}h{(secs_remain % 3600)/60:.0f}m to go, finish at {end_time:%a %d %b %H:%M:%S}')

    
    def suggest(self):
        games = pd.concat(list(self.games.values()))
        parts = games.index.str.extract(r'(?P<run>.*)\.(?P<idx>.*)')
        parts['idx'] = parts['idx'].astype(int)
        parts['is_last'] = parts.groupby('run').apply(lambda df: df.idx == df.idx.max()).reset_index(level=0, drop=True)
        parts.index = games.index

        succ = parts.run + '.' + (parts.idx + 1).astype(str)
        succ = succ.index.values[:, None] == succ.values[None, :]
        succ = succ | succ.T

        first = (parts.idx.values[:, None] == 0) & (parts.idx.values[None, :] == 0)
        last = parts.is_last.values[:, None] & parts.is_last.values[None, :]

        targets = succ | first | last

        sugg = ((games == 0) & (targets > 0)).stack().loc[lambda df: df]

        if len(sugg):
            self.report(sugg)
            return sugg.sample(1).index[0]

class FullSuggester:

    def __init__(self, snaps):
        self.games, self.wins = {}, {}
        for b in snaps.boardsize.unique():
            self.games[b], self.wins[b] = data.load(b, snaps[snaps.boardsize == b].index)

        self.start = time.time()
        self.init_matches = self.played()
        self.moves = 0
    
    def played(self):
        return sum(g.gt(0).sum().sum() for g in self.games.values())

    def remaining(self):
        return sum(g.eq(0).sum().sum() for g in self.games.values())

    def report(self):
        matches_played = self.played() - self.init_matches
        time_passed = (time.time() - self.start)
        match_rate = (matches_played+1)/time_passed
        move_rate = self.moves/time_passed

        matches_remain = self.remaining()
        time_remain = pd.to_timedelta(matches_remain/match_rate, unit='s')
        end_time = pd.Timestamp.now() + time_remain

        log.info(f'{60*match_rate:.0f} matches/min, {move_rate:.0f} moves/sec. {time_remain.total_seconds()/3600:.0f}hrs to go, finish at {end_time:%a %d %b %H:%M:%S}')

    def update(self, results, boardsize):
        self.games[boardsize], self.wins[boardsize] = update(self.games[boardsize], self.wins[boardsize], results)
        self.moves += sum(r.moves for r in results)
        
        self.report()

    def suggest(self):
        suggs = pd.concat([(g == 0).stack().loc[lambda df: df] for b, g in self.games.items()])
        
        if len(suggs):
            return suggs.sample(1).index[0]
        else:
            log.info('No suggestions')

def structured_eval(n_workers=12, suggester=FullSuggester, **kwargs):
    # ```!while true; do python -c "from grid.eval import *; structured_eval()" || true; done```

    #TODO: Unify these eval fns
    snaps = data.snapshot_solns(**kwargs, solve=False) 

    compile(snaps.index[0])

    suggester = FullSuggester(snaps)

    futures = {}
    with DeviceExecutor(n_workers, initializer=init) as pool:
        while True:
            for key, future in list(futures.items()):
                if future.done():
                    results, boardsize = future.result()
                    suggester.update(results, boardsize)
                    del futures[key]
                    data.save(boardsize, suggester.games[boardsize], suggester.wins[boardsize])

            while len(futures) < n_workers:
                sugg = suggester.suggest()
                if sugg:
                    futures[(np.random.randint(2**32), *sugg)] = pool.submit(evaluate, *sugg)
                else:
                    break

            if len(futures) == 0:
                log.info('Finished')
                break

            time.sleep(.1)

def fast_eval(boardsize=5, n_workers=2, **kwargs):
    from boardlaw.arena import multi, common

    snaps = data.snapshot_solns(boardsize, solve=False)

    raw = asymdata.pandas(boardsize)
    raw['games'] = raw.black_wins + raw.white_wins

    games = raw.games.unstack().reindex(index=snaps.index, columns=snaps.index).fillna(0)

    def worldfunc(n_envs):
        return common.worlds(snaps.run.iloc[0], n_envs, device='cuda')

    def agentfunc(name):
        row = snaps.loc[name]
        return common.agent(row.run, row.idx, device='cuda')

    from IPython import display

    for rs, stats in multi.evaluate(worldfunc, agentfunc, games, chunksize=64, n_workers=n_workers):
        asymdata.save(rs)

        display.clear_output(wait=True)
        multi.print_stats(boardsize, stats)

