import time
import numpy as np
import torch
from rebar import arrdict, recording
from pavlov import runs, storage
from logging import getLogger
from . import arena

log = getLogger(__name__)

def combine_actions(decisions, masks):
    actions = torch.cat([d.actions for d in decisions.values()])
    for mask, decision in zip(masks.values(), decisions.values()):
        actions[mask] = decision.actions
    return actions

def expand(exemplar, n_envs):
    if exemplar.dtype in (torch.half, torch.float, torch.double):
        default = np.nan
    elif exemplar.dtype in (torch.short, torch.int, torch.long):
        default = -1
    else:
        raise ValueError('Don\'t have a default for "{exemplar.dtype}"')
    shape = (n_envs, *exemplar.shape[1:])
    return torch.full(shape, default, dtype=exemplar.dtype, device=exemplar.device)
    
def combine_decisions(dtrace, mtrace):
    agents = {a for d in dtrace for a in d}
    n_envs = next(iter(mtrace[0].values())).size(0)
    results = arrdict.arrdict()
    for a in agents:
        exemplar = [d[a] for d in dtrace if a in d][0]
        device = next(iter(arrdict.leaves(exemplar))).device

        a_results = []
        for d, m in zip(dtrace, mtrace):
            expanded = exemplar.map(expand, n_envs=n_envs)
            if a in m:
                expanded[m[a]] = d[a]
                expanded['mask'] = m[a]
            else:
                expanded['mask'] = torch.zeros((n_envs,), dtype=bool, device=device)
            a_results.append(expanded)
        results[str(a)] = arrdict.stack(a_results)
    return results

@torch.no_grad()
def rollout(worlds, agents, n_steps=None, n_trajs=None, n_reps=None, **kwargs):
    assert sum(x is not None for x in (n_steps, n_trajs, n_reps)) == 1, 'Must specify exactly one of n_steps or n_trajs or n_reps'

    trace, dtrace, mtrace = [], [], []
    steps, trajs = 0, 0
    reps = torch.zeros(worlds.n_envs, device=worlds.device)
    while True:
        decisions, masks = {}, {}
        for i, agent in enumerate(agents):
            mask = worlds.seats == i
            if mask.any():
                decisions[i] = agent(worlds[mask], **kwargs)
                masks[i] = mask

        actions = combine_actions(decisions, masks)
        
        worlds, transitions = worlds.step(actions)
        trace.append(arrdict.arrdict(
            actions=actions,
            transitions=transitions,
            worlds=worlds))
        
        mtrace.append(masks)
        dtrace.append(decisions)

        steps += 1

        if n_steps and (steps >= n_steps):
            break
        trajs += transitions.terminal.sum()
        if n_trajs and (trajs >= n_trajs):
            break
        reps += transitions.terminal
        if n_reps and (reps >= n_reps).all():
            break

    trace = arrdict.stack(trace)
    trace['decisions'] = combine_decisions(dtrace, mtrace)

    return trace

def plot_all(f):

    def proxy(state):
        import numpy as np
        import matplotlib.pyplot as plt

        B = state.seats.shape[0]
        assert B < 65, f'Plotting {B} traces will be prohibitively slow' 
        n_rows = int(B**.5)
        n_cols = int(np.ceil(B/n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)

        for e in range(B):
            f(state, e, ax=axes.flatten()[e])
        
        return fig
    return proxy

def record_worlds(worlds, N=0):
    state = arrdict.numpyify(worlds)
    with recording.ParallelEncoder(plot_all(worlds.plot_worlds), N=N, fps=1) as encoder:
        for i in range(state.board.shape[0]):
            encoder(state[i])
    return encoder
    
def record(world, agents, N=0, **kwargs):
    trace = rollout(world, agents, **kwargs)
    return record_worlds(trace.worlds, N=N)

def rollout_model(run=-1, mohex=True, eval=True, n_envs=1):
    from boardlaw import mcts, hex
    boardsize = runs.info(run)['boardsize']
    worlds = hex.Hex.initial(n_envs=n_envs, boardsize=boardsize)
    network = storage.load_raw(run, 'model')
    agent = mcts.MCTSAgent(network, n_nodes=64)
    if mohex:
        from boardlaw import mohex
        agents = [agent, mohex.MoHexAgent(solver=True)]
    else:
        agents = [agent, agent]
    return rollout(worlds, agents, n_reps=1, eval=eval)

def compare(fst_run=-1, snd_run=-1, n_envs=256, device='cuda:1'):
    import pandas as pd
    from .main import worldfunc, agentfunc

    world = worldfunc(n_envs, device=device)

    fst = agentfunc(device=device)
    fst.load_state_dict(storing.select(storing.load_latest(fst_run), 'agent'))

    snd = agentfunc(device=device)
    snd.load_state_dict(storing.select(storing.load_latest(snd_run), 'agent'))

    bw = rollout(world, [fst, snd], n_reps=1)
    bw_wins = (bw.transitions.rewards[bw.transitions.terminal.cumsum(0) <= 1] == 1).sum(0)

    wb = rollout(world, [snd, fst], n_reps=1)
    wb_wins = (wb.transitions.rewards[wb.transitions.terminal.cumsum(0) <= 1] == 1).sum(0)

    # Rows: black, white; cols: old, new
    wins = torch.stack([bw_wins, wb_wins.flipud()]).detach().cpu().numpy()

    return pd.DataFrame(wins/n_envs, ['black', 'white'], ['fst', 'snd'])

def load(run):
    from boardlaw import mohex, hex, mcts
    boardsize = runs.info(run)['params']['boardsize']

    worlds = hex.Hex.initial(9, boardsize=boardsize)

    network = storage.load_raw(run, 'model')
    agent = mcts.MCTSAgent(network)
    agent.load_state_dict(storage.load_latest(run)['agent'])

    return worlds, agent


def demo_record(run=-1):
    from boardlaw import analysis

    world = arena.common.worlds(run, 1)
    agent = arena.common.agent(run)
    return analysis.record(world, [agent, agent], n_trajs=1).notebook()

def demo_rollout(run=-1):
    from boardlaw import mohex, analysis
    mhx = mohex.MoHexAgent()

    worlds, agent = load(run)
    trace = rollout(worlds, [agent, agent], n_reps=1)

    trace.transitions.rewards.sum(0).sum(0)

def board_runs(boardsize=9):
    import pandas as pd
    from pavlov import stats, runs
    import matplotlib.pyplot as plt

    # date of the first 9x9 run
    valid = runs.pandas().query(f'_created > "2020-12-23 09:52Z" & boardsize == {boardsize} & parent == ""')

    results = {}
    for name in valid.index:
        if stats.exists(name, 'elo-mohex'):
            s = stats.pandas(name, 'elo-mohex')
            if len(s) > 60 and (s.notnull().sum() > 15).any():
                results[name] = s.μ
    df = pd.concat(results, 1)
    smoothed = df.ffill(limit=3).where(df.bfill().notnull()).iloc[3:].head(900)

    with plt.style.context('seaborn-poster'):
        ax = smoothed.plot(cmap='viridis_r', legend=False, linewidth=1.5)
        ax.set_facecolor('whitesmoke')
        ax.grid(axis='y')
        ax.set_ylim(None, 0)
        ax.set_ylabel('eElo')
        ax.set_title(f'all runs on {boardsize}x{boardsize} boards')

    return smoothed

def mohex_benchmark(run):
    from boardlaw import mohex
    from boardlaw.main import worldfunc, agentfunc
    from boardlaw.arena import evaluate

    n_envs = 8
    worlds = worldfunc(n_envs)
    agent = agentfunc()
    agent.load_state_dict(storage.load_latest('*gross-steams')['agent'])
    # agent.kwargs['n_nodes'] = 512
    # agent.kwargs['noise_eps'] = 0.
    mhx = mohex.MoHexAgent()

    return evaluate(worlds, {'boardlaw': agent, 'mohex': mhx})

def snapshot_kl_divs(run):
    import pandas as pd
    from pavlov import runs, storage
    from boardlaw import hex
    from boardlaw.main import mix
    import torch
    from tqdm.auto import tqdm

    m = storage.load_raw(run, 'model')
    worlds = mix(hex.Hex.initial(n_envs=16*1024, boardsize=runs.info(run)['params']['boardsize']))

    logits = {}
    for idx in tqdm(storage.snapshots(run)):
        sd = storage.load_snapshot(run, idx)['agent']
        m.load_state_dict(storage.expand(sd)['network'])
        logits[idx] = m(worlds).logits.detach()
        
    kldivs = {}
    for i in logits:
        for j in logits:
            li = logits[i]
            lj = logits[j]
            terms = -li.exp().mul(lj - li)
            mask = torch.isfinite(terms)
            kldiv = terms.where(mask, torch.zeros_like(terms)).sum(-1)
            kldivs[i, j] = kldiv.mean().item()
    df = pd.Series(kldivs).unstack()

    return df