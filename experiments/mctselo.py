import torch
from pavlov import storage, runs
from boardlaw import hex, mohex, mcts, analysis

def count_wins(transitions):
    return (transitions.rewards
                .where(transitions.terminal[..., None].cumsum(0) <= 1, torch.zeros_like(transitions.rewards))
                .flatten(0, 1)
                .eq(1)
                .sum(0))

def test(n_nodes):
    run = '*great-fits'
    boardsize = runs.info(run)['boardsize']
    worlds = hex.Hex.initial(n_envs=256, boardsize=boardsize)
    network = storage.load_raw(run, 'model')
    sd = storage.load_snapshot(run, n=4)['agent']
    network.load_state_dict(storage.expand(sd)['network'])
    A = mcts.MCTSAgent(network, n_nodes=n_nodes)
    B = mcts.DummyAgent(network)

    fst = analysis.rollout(worlds, [A, B], n_reps=1, eval=False)
    snd = analysis.rollout(worlds, [B, A], n_reps=1, eval=False)

    wins = count_wins(fst.transitions) + count_wins(snd.transitions).flipud()

    rate = wins[0]/wins.sum()
    elo = torch.log(rate) - torch.log(1 - rate)
    return elo.item()
