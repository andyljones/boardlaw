import statsmodels.formula.api as smf
import plotnine as pn
import pandas as pd
from . import eval, plot, asymdata, data
import numpy as np
import time

def node_eval(boardsize, nodes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], n_workers=4):
    from boardlaw.arena import multi, common
    n_envs_per = 512

    snaps = data.snapshot_solns(boardsize, solve=False)
    if 'nodes' in snaps:
        snaps = snaps[snaps.nodes.isnull()]

    raw = asymdata.pandas(boardsize)
    raw['games'] = raw.black_wins + raw.white_wins

    games = raw.games.unstack()

    def worldfunc(n_envs):
        return common.worlds(snaps.run.iloc[0], n_envs, device='cuda')

    def agentfunc(name):
        shortrun, idx, nodes = name.split('.')
        row = snaps.loc[f'{shortrun}.{idx}']
        agent = common.agent(row.run, row.idx, device='cuda')
        agent.kwargs['n_nodes'] = int(nodes)
        return agent
        
    from IPython import display

    jobs = {}
    for nickname in snaps.index:
        names = [f'{nickname}.{n}' for n in nodes]
        subgames = games.reindex(index=names, columns=names).fillna(0)
        subgames.values[np.diag_indices_from(subgames)] = n_envs_per
        jobs[nickname] = subgames
        
    from boardlaw.arena import multi
    from rebar import parallel
    from random import shuffle

    multi.set_start_method('spawn', True)
    stats = multi.initial_stats(len(jobs))
    with parallel.parallel(multi.evaluate_chunk, N=n_workers, executor='cuda') as pool:
        keys = list(jobs)
        shuffle(keys)

        jobs = {k: pool(worldfunc, agentfunc, jobs[k], n_envs_per) for k in keys}
        while jobs:
            for k, future in list(jobs.items()):
                if future.done():
                    results = future.result()
                    multi.update_stats(stats, results)
                    asymdata.save(results)
                    del jobs[k]

            stats['end'] = time.time()
            #yield [], stats.copy()

            display.clear_output(wait=True)
            multi.print_stats(boardsize, stats)
            time.sleep(.1)


def summarize(boardsize=9):
    snaps = data.snapshot_solns(boardsize, solve=False)
    raw = asymdata.pandas(boardsize).reset_index()
    raw['games'] = raw.black_wins + raw.white_wins

    regex = r'(?P<run>[\w-]+)\.(?P<idx>\d+)(?:\.(?P<nodes>\d+))?'
    black_spec = raw.black_name.str.extract(regex).rename(columns=lambda c: 'black_' + c).fillna(64)
    white_spec = raw.white_name.str.extract(regex).rename(columns=lambda c: 'white_' + c).fillna(64)
    raw = pd.concat([raw, black_spec, white_spec], 1)
    raw['black_nickname'] = raw.black_run + '.' + raw.black_idx
    raw['white_nickname'] = raw.white_run + '.' + raw.white_idx
    raw['black_nodes'] = raw.black_nodes.astype(float)
    raw['white_nodes'] = raw.white_nodes.astype(float)

    raw = raw.groupby(['black_nickname', 'black_nodes', 'white_nickname', 'white_nodes']).last().reset_index()
    raw['black_name'] = raw.black_nickname + '.' + raw.black_nodes.astype(int).astype(str)
    raw['white_name'] = raw.white_nickname + '.' + raw.white_nodes.astype(int).astype(str)    

    df = raw.pivot('black_name', 'white_name', ['black_wins', 'white_wins'])
    wins = (df.black_wins + df.white_wins.T)
    games = wins + wins.T

    elos = asymdata.fast_elos(wins, games)
    elos = pd.Series(elos, wins.index)

    regex = r'(?P<run>[\w-]+)\.(?P<idx>\d+)(?:\.(?P<nodes>\d+))?'
    info = elos.index.str.extract(regex)
    info['nickname'] = info.run + '.' + info.idx
    info['nodes'] = info['nodes'].astype(float)
    info['elo'] = elos.values
    info = pd.merge(info[['nodes', 'nickname', 'elo']], snaps, left_on='nickname', right_on='nickname')    

    info['test_flops'] = info.nodes*info.flops/info.samples
    # Round the train flops so that the frontier is smooth
    info['train_flops'] = info['flops'].pipe(np.log10).round(1).pipe(lambda s: 10**s) 

    return info

def plot_train_test_frontiers(info):
    info = info[info.nodes > 1]

    frontiers = {}
    for e in np.linspace(-10, 0, 11):
        frontiers[e] = info[info.elo > e].groupby('train_flops').test_flops.min().expanding().min()
        
    frontiers = pd.concat(frontiers).unstack().T

    distinct = frontiers.pipe(np.log10).round(1).pipe(lambda df: 10**df)
    distinct = distinct.where(distinct.iloc[-1].eq(distinct).cumsum().le(1))
    distinct = distinct.stack().reset_index()
    distinct.columns = ['train_flops', 'elo', 'test_flops']

    model = smf.ols('np.log10(test_flops) ~ np.log10(train_flops) + C(elo) + 1', distinct).fit()
    distinct['test_flops_hat'] = 10**model.predict(distinct)

    df = frontiers.stack().reset_index()
    df.columns = ['train_flops', 'elo', 'test_flops']
    df = pd.merge(df, distinct.drop('test_flops', 1), on=['train_flops', 'elo'], how='left')

    ps = model.params.copy()
    ps = ps.apply(lambda x: f'{float(f"{x:.3g}"):g}')
    s = f'$\ln_{{10}}\mathrm{{test}} = {ps["np.log10(train_flops)"]} \cdot \ln_{{10}}\mathrm{{train}} + C$'

    return (pn.ggplot(df, pn.aes(x='train_flops', color='factor(elo)'))
        + pn.geom_line(pn.aes(y='test_flops'), size=2)
        + pn.geom_line(pn.aes(y='test_flops_hat'), size=2, linetype='dashed')
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_y_continuous(trans='log10')
        + pn.annotate('text', 3e13, 1.5e9, label=s, size=20)
        + pn.scale_color_discrete(name='elo')
        + pn.labs(title='tradeoff between train and test compute is roughly even')
        + plot.mpl_theme()
        + plot.poster_sizes())

def plot_test_perf(info):
    finals = info.sort_values('idx').groupby(['run', 'nodes']).last().reset_index()
    finals['arch'] = finals['width'].astype(str) + '/' + finals['depth'].astype(str)

    return (pn.ggplot(finals)
        + pn.geom_point(pn.aes(x='test_flops', y='400/np.log(10)*elo', color='params', label='arch'))
        + pn.geom_line(pn.aes(x='test_flops', y='400/np.log(10)*elo', color='params', label='arch', group='run'))
        + pn.scale_x_continuous(trans='log10')
        + pn.scale_color_continuous(trans='log10')
        + pn.labs(title='test-flops-v-perf tradeoff, one snapshot per line', y='elo')
        + plot.no_colorbar_ticks()
        + plot.mpl_theme()
        + plot.poster_sizes())