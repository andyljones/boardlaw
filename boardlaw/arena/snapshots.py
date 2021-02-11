import time
import pandas as pd
from pavlov import storage, runs
from . import common, evaluator, database
from .. import mcts
from logging import getLogger

log = getLogger(__name__)

def snapshot_agents(run, agentfunc, **kwargs):
    if not isinstance(run, (int, str)):
        agents = {}
        for r in run:
            agents.update(snapshot_agents(r, agentfunc, **kwargs))
        return agents

    period = kwargs.get('period', 1)
    tail = kwargs.get('tail', int(1e6))
    try:
        stored = pd.DataFrame.from_dict(storage.snapshots(run), orient='index').tail(tail).iloc[::period]
    except ValueError:
        return {}
    else:
        agents = {} 
        for idx, info in stored.iterrows():
            if idx % period == 0:
                name = pd.Timestamp(info['_created']).strftime(r'%y%m%d-%H%M%S-snapshot')
                agents[name] = common.agent(run, idx, device='cuda')
        return agents

def matchups(run=-1, count=1, **kwargs):

    run = runs.resolve(run)
    agentfunc = lambda: mcts.MCTSAgent(storage.load_raw(run, 'model'))
    agents = snapshot_agents(run, agentfunc, **kwargs)
    worlds = common.worlds(run, 256, device='cuda')

    while True:
        agents = snapshot_agents(run, agentfunc, **kwargs)

        n, w = database.symmetric(run, agents)
        zeros = (n
            .stack()
            .loc[lambda s: s < count]
            .reset_index()
            .loc[lambda df: df.black_name != df.white_name])

        indices = {n: i for i, n in enumerate(n.index)}
        diff = abs(zeros.black_name.replace(indices) - zeros.white_name.replace(indices))
        ordered = zeros.loc[diff.sort_values().index]
        # Sample so there's no problems if we run in parallel
        if len(ordered) == 0:
            log.info('No matchups to play')
            time.sleep(15)
            continue
        matchup = ordered.head(10).sample(1).iloc[0, :2].tolist()

        log.info(f'Playing {matchup}')
        matchup = {m: agents[m] for m in matchup}
        results = evaluator.evaluate(worlds, matchup)

        wins, games = int(results[0].wins[0] + results[1].wins[1]), int(sum(r.games for r in results))
        log.info(f'Storing. {wins} wins in {games} games for {list(matchup)[0]} ')
        database.save(run, results)