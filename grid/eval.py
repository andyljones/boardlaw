import plotnine as pn
import scipy as sp
import numpy as np
import activelo
import pandas as pd
from boardlaw import arena
from pavlov import storage, runs
from rebar import dotdict
from IPython import display
from concurrent.futures import ProcessPoolExecutor

def snapshots(boardsize):
    snapshots = {}
    for r in runs.runs(description=f'bee/{boardsize}'):
        for i, s in storage.snapshots(r).items():
            snapshots[r, i] = s
    return (pd.DataFrame.from_dict(snapshots, orient='index')
                    .rename_axis(index=('run', 'idx'))
                    .reset_index())

def parameters(snaps):
    params = {}
    for idx, row in snaps.iterrows():
        s = storage.load_snapshot(row.run, row.idx)
        params[idx] = {**runs.info(row.run)['params'], 'samples': s['n_samples'], 'flops': s['n_flops']}
    return pd.DataFrame.from_dict(params, orient='index')

def evaluate(Aname, Bname):
    Arun, Aidx = Aname.split('.')
    Brun, Bidx = Aname.split('.')
    A = arena.common.agent(f'*{Arun}', int(Aidx), 'cuda')
    B = arena.common.agent(f'*{Brun}', int(Bidx), 'cuda')
    worlds = arena.common.worlds(f'*{Arun}', 512, 'cuda')

    return arena.common.evaluate(worlds, {Aname: A, Bname: B})

def update(games, wins, results):
    games, wins = games.copy(), wins.copy()
    for result in results:
        games.loc[result.names[0], result.names[1]] += result.games
        games.loc[result.names[1], result.names[0]] += result.games
        wins.loc[result.names[0], result.names[1]] += result.wins[0]
        wins.loc[result.names[1], result.names[0]] += result.wins[1]
    return games, wins

def report(soln):
    μ, σ = arena.analysis.difference(soln, soln.μ.idxmin())

    display.clear_output(wait=True)
    print(f'nonzero: {(soln.μ != 0).mean():.0%}')
    print(f'μ_max: {μ.max():.1f}')
    print(f'σ_ms: {σ.pow(2).mean()**.5:.2f}')

def suggest(soln, temp=1):
    imp = activelo.improvement(soln)

    logits = temp*(imp.values.flatten() - sp.special.logsumexp(imp.values.flatten()))

    n_agents = imp.shape[0]
    choice = np.random.choice(np.arange(logits.size), p=np.exp(logits))
    sugg = (imp.index[choice // n_agents], imp.index[choice % n_agents])

    return sugg

def params(df):
    intake = (df.boardsize**2 + 1)*df.width
    body = (df.width**2 + df.width) * df.depth
    output = df.boardsize**2 * (df.width + 1)
    return intake + body + output

def plot(snaps):
    (pn.ggplot(data=snaps)
        + pn.geom_line(pn.aes(x='flops', y='μ', group='run', color='params'))
        + pn.scale_x_continuous(trans='log10'))

def run(boardsize=3, n_workers=8):
    snaps = snapshots(boardsize)
    snaps = pd.concat([snaps, parameters(snaps)], 1)
    snaps['nickname'] = snaps.run.str.extract('.* (.*)', expand=False) + '.' + snaps.idx.astype(str)
    snaps['params'] = params(snaps)
    snaps = snaps.set_index('nickname')

    n_agents = len(snaps)
    wins  = pd.DataFrame(np.zeros((n_agents, n_agents)), snaps.index, snaps.index)
    games = pd.DataFrame(np.zeros((n_agents, n_agents)), snaps.index, snaps.index)

    soln = None
    futures = {}
    with ProcessPoolExecutor(n_workers) as pool:
        while True:
            if len(futures) < n_workers:
                soln = activelo.solve(games, wins, soln=soln)
                sugg = suggest(soln)
                futures[sugg] = pool.submit(evaluate, *sugg)

            for key, future in list(futures.items()):
                if future.done():
                    results = future.result()
                    games, wins = update(games, wins, results)
                    del futures[key]
        
            if soln is not None:
                report(soln)
                _, σ = arena.analysis.difference(soln, soln.μ.idxmin())
                if σ.pow(2).mean()**.5 < .1:
                    break

    snaps['μ'], snaps['σ'] = arena.analysis.difference(soln, soln.μ.idxmin())
