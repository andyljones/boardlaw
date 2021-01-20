import numpy as np
import pandas as pd
import torch
from pavlov import storage, runs
from boardlaw import hex, mohex, mcts, analysis

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
    return {'elo': elo.item(), 'kl': kl.item()}

def run(source_run):
    results = []
    for c in [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2]:
        for n in [2, 4, 8, 16, 32, 64, 128]:
            results.append({
                'c_puct': c,
                'n_nodes': n,
                **test(source_run, 1, c_puct=c, n_nodes=n)})
            print(results[-1])
    df = pd.DataFrame(results).pivot('c_puct', 'n_nodes', ['elo', 'kl'])
    return df