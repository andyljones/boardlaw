import numpy as np
import pandas as pd
import torch
from pavlov import storage, runs
from boardlaw import hex, mohex, mcts, analysis
import matplotlib.pyplot as plt

def count_wins(transitions):
    return (transitions.rewards
                .where(transitions.terminal[..., None].cumsum(0) <= 1, torch.zeros_like(transitions.rewards))
                .flatten(0, 1)
                .eq(1)
                .sum(0))

def kl_div(decisions):
    mask = decisions.mask[..., None] & (decisions.logits > -np.inf) & (decisions.prior > -np.inf)
    zeros = torch.zeros_like(decisions.logits)
    logits = decisions.logits.where(mask, zeros).float()
    prior = decisions.prior.where(mask, zeros).float()
    return (logits - prior).mul(logits.exp()).sum(-1).mean()

def rel_entropy(decisions):
    mask = decisions.mask[..., None] & (decisions.logits > -np.inf) & (decisions.prior > -np.inf)
    zeros = torch.zeros_like(decisions.logits)
    logits = decisions.logits.where(mask, zeros).float()
    ent = (logits.exp()*logits).sum(-1)
    norm = mask.float().sum(-1).log()
    rel_ent = ent/norm
    rel_ent[norm == 0.] = 0.
    return -rel_ent.mean()

def test(run, snapshot=-1, **kwargs):
    boardsize = runs.info(run)['boardsize']
    worlds = hex.Hex.initial(n_envs=1024, boardsize=boardsize)

    network = storage.load_raw(run, 'model')
    sd = storage.load_snapshot(run, n=snapshot)['agent']
    network.load_state_dict(storage.expand(sd)['network'])
    A = mcts.MCTSAgent(network.cuda(), **kwargs)

    network = storage.load_raw(run, 'model')
    sd = storage.load_snapshot(run, n=snapshot)['agent']
    network.load_state_dict(storage.expand(sd)['network'])
    B = mcts.DummyAgent(network.cuda())

    fst = analysis.rollout(worlds, [A, B], n_reps=1, eval=False)
    snd = analysis.rollout(worlds, [B, A], n_reps=1, eval=False)

    wins = count_wins(fst.transitions) + count_wins(snd.transitions).flipud()

    rate = wins[0]/wins.sum()
    elo = torch.log(rate) - torch.log(1 - rate)

    kl =  (kl_div(fst.decisions['0']) + kl_div(snd.decisions['0']))/2
    ent =  (rel_entropy(fst.decisions['0']) + rel_entropy(snd.decisions['0']))/2
    return {'elo': elo.item(), 'kl': kl.item(), 'ent': ent.item()}

def run(source_run, snapshot):
    results = []
    for n in [2, 4, 8, 16, 32, 64, 128]:
        for c in [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2]:
            results.append({
                'c_puct': c,
                'n_nodes': n,
                **test(source_run, snapshot, c_puct=c, n_nodes=n)})
            print(results[-1])
    df = pd.DataFrame(results).pivot('c_puct', 'n_nodes', ['elo', 'kl', 'ent'])
    return df

def as_dataframe(s):
    from io import StringIO
    df = pd.read_csv(StringIO(s), sep='\t', index_col=0).iloc[1:]
    df.index = pd.to_numeric(df.index)
    df.columns.name = 'n_nodes'
    df.index.name = 'c_puct'
    return df

def plot(df, ax=None):
    _, ax = plt.subplots() if ax is None else (None, ax)
    ax = df.plot(cmap='viridis', marker='o', ax=ax)
    ax.set_xscale('log', basex=2)
    ax.axhline(0, color='k', alpha=.5)
    return ax