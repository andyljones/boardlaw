import pandas as pd
from . import database, evaluator, analysis
import activelo
from logging import getLogger
from pavlov import stats

log = getLogger(__name__)

def snapshot_trial(run_name, worlds, agents):
    n, w = database.symmetric(run_name, agents)
    log.info(f'Loaded {int(n.sum().sum())} games')

    valid = n.index[n.index.str.endswith('snapshot')]
    if len(valid) < 2:
        raise ValueError('Need at least two periodic agents for a periodic step')
    n = n.reindex(index=valid, columns=valid)
    w = w.reindex(index=valid, columns=valid)

    soln = activelo.solve(n.values, w.values)
    log.info(f'Fitted a posterior, {(soln.σd**2).mean()**.5:.2f}σd over {n.shape[0]} agents')
    matchup = activelo.suggest(soln)
    matchup = [n.index[m] for m in matchup]

    agents = {m: agents[m] for m in matchup}
    log.info('Playing ' + ' v. '.join(agents))
    results = evaluator.evaluate(worlds, agents)

    wins, games = int(results[0].wins[0] + results[1].wins[1]), int(sum(r.games for r in results))
    log.info(f'Storing. {wins} wins in {games} games for {list(agents)[0]} ')
    database.save(run_name, results)

def mohex_trial(run_name, worlds, agents):
    n, w = database.symmetric(run_name, agents)
    log.info(f'Loaded {int(n.sum().sum())} games')

    valid = n.index[n.index.str.endswith('periodic') | (n.index == 'mohex')]
    if len(valid) < 2:
        raise ValueError('Need at least two periodic agents for a periodic step')
    n = n.reindex(index=valid, columns=valid)
    w = w.reindex(index=valid, columns=valid)

    soln = activelo.solve(n.values, w.values)
    log.info(f'Fitted a posterior, {(soln.σd**2).mean()**.5:.2f}σd over {n.shape[0]} agents')
    improvement = activelo.improvement(soln)
    improvement = pd.DataFrame(improvement, n.index, n.columns).loc['mohex'].drop('mohex')
    matchup = ('mohex', improvement.idxmax())

    agents = {m: agents[m] for m in matchup}
    log.info('Playing ' + ' v. '.join(agents))
    results = evaluator.evaluate(worlds, agents)

    wins = int(results[0].wins[0] + results[1].wins[1])
    games = int(sum(r.games for r in results))
    moves = int(sum(r.moves for r in results))
    log.info(f'Storing. {wins} wins in {games} games for {list(agents)[0]}; {moves/(2*games):.1f} average moves per player')
    stats.mean('moves-mohex', moves, 2*games)
    database.save(run_name, results)

_latest_matchup = None
_latest_results = []
def latest_trial(run_name, worlds, agents):
    n, w = database.symmetric(run_name, agents)
    log.info(f'Loaded {int(n.sum().sum())} games')

    [latest] = n.index[n.index.str.endswith('latest')]
    first_periodic = n.index[n.index.str.endswith('periodic')][0]
    latest_periodic = n.index[n.index.str.endswith('periodic')][-1]
    matchup = (latest, latest_periodic)

    agents = {m: agents[m] for m in matchup}
    log.info('Playing ' + ' v. '.join(agents))
    results = evaluator.evaluate(worlds, agents)

    global _latest_matchup, _latest_results
    if _latest_matchup != matchup:
        _latest_matchup = matchup
        _latest_results = []
    _latest_results.extend(results)

    log.info(f'Got {len(_latest_results)} results to accumulate')
    for r in _latest_results:
        w.loc[r.names[0], r.names[1]] += r.wins[0]
        w.loc[r.names[1], r.names[0]] += r.wins[1]
        n.loc[r.names[0], r.names[1]] += r.games
        n.loc[r.names[1], r.names[0]] += r.games
    
    wins, games = int(w.loc[matchup[0], matchup[1]]), int(n.loc[matchup[0], matchup[1]])
    log.info(f'Fitting posterior. {wins} wins for {list(agents)[0]} in {games} games')
    soln = activelo.solve(n.values, w.values)
    log.info(f'Fitted posterior, {(soln.σd**2).mean()**.5:.2f}σd over {n.shape[0]} agents')
    μ = pd.Series(soln.μ, n.index)

    soln = analysis.pandas(soln, n.index)
    μm, σm = analysis.difference(soln, 'mohex', latest)
    stats.mean_std('elo-mohex/latest', μm, σm)
    μ0, σ0 = analysis.difference(soln, first_periodic, latest)
    stats.mean_std('elo-first/latest', μ0, σ0)
    log.info(f'eElo for {latest} is {μ0:.2f}±{2*σ0:.2f} v. the first agent, {μm:.2f}±{2*σm:.2f} v. mohex')

    μm, σm = analysis.difference(soln, 'mohex', latest_periodic)
    stats.mean_std('elo-mohex/periodic', μm, σm)
    if latest_periodic != first_periodic:
        μ0, σ0 = analysis.difference(soln, first_periodic, latest_periodic)
        stats.mean_std('elo-first/periodic', μ0, σ0)
        log.info(f'eElo for {latest_periodic} is {μ0:.2f}±{2*σ0:.2f} v. the first agent, {μm:.2f}±{2*σm:.2f} v. mohex')
    else:
        log.info(f'eElo for {latest_periodic} is {μm:.2f}±{2*σm:.2f} v. mohex')

def trial(run_name, worlds, agents, kind):
    log.info(f'Running a "{kind}" step')
    try:
        globals()[f'{kind}_trial'](run_name, worlds, agents)
    except Exception as e:
        log.error(f'Failed while running a "{kind}" step with a "{e}" error')

