import torch
import torch.testing
import torch.distributions
import pytest
from . import cuda
from rebar import arrdict
import aljpy

### ROOT TESTS

def test_root_one_node():
    data = arrdict.arrdict(
        logits=torch.tensor([[1/3, 2/3]]).log(),
        w=torch.tensor([[0.]]),
        n=torch.tensor([0]),
        c_puct=torch.tensor(1.),
        seats=torch.tensor([0]),
        terminal=torch.tensor([False]),
        children=torch.tensor([[-1, -1]]))
    
    expected = torch.tensor([[1/3, 2/3]]).cuda() 
    m = cuda.mcts(**data.cuda()[None])
    actual = cuda.root(m)
    torch.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-3)

### DESCEND TESTS

def assert_distribution(xs, freqs):
    for i, freq in enumerate(freqs):
        actual = (xs == i).float().mean()
        ci = 3*(freq*(1-freq)/len(xs))**.5
        assert abs(actual - freq) <= ci, f'Expected {freq:.2f}±{ci:.2f} to be {i}, got {actual:.2f}'

def test_one_node():
    data = arrdict.arrdict(
        logits=torch.tensor([[1/3, 2/3]]).log(),
        w=torch.tensor([[0.]]),
        n=torch.tensor([0]),
        c_puct=torch.tensor(1.),
        seats=torch.tensor([0]),
        terminal=torch.tensor([False]),
        children=torch.tensor([[-1, -1]]))
    
    m = cuda.mcts(**data.cuda()[None].repeat_interleave(1024, 0))
    result = cuda.descend(m)
    assert_distribution(result.parents, [1])
    assert_distribution(result.actions, [1/3, 2/3])

def test_high_cpuct():
    # Ignore the high-q path, stay close to the prior
    data = arrdict.arrdict(
        logits=torch.tensor([
            [1/3, 2/3],
            [1/4, 3/4],
            [1/5, 4/5]]).log(),
        w=torch.tensor([[0.], [0.], [1.,]]),
        n=torch.tensor([2, 1, 1]),
        c_puct=torch.tensor(1000.),
        seats=torch.tensor([0, 0, 0]),
        terminal=torch.tensor([False, False, False]),
        children=torch.tensor([
            [1, 2], 
            [-1, -1], 
            [-1, -1]]))

    m = cuda.mcts(**data.cuda()[None].repeat_interleave(1024, 0))
    result = cuda.descend(m)
    assert_distribution(result.parents, [0, 1/3, 2/3])
    assert_distribution(result.actions, [1/3*1/4 + 2/3*1/5, 1/3*3/4 + 2/3*4/5])

def test_low_cpuct():
    # Concentrate on the high-q path
    data = arrdict.arrdict(
        logits=torch.tensor([
            [1/3, 2/3],
            [1/4, 3/4],
            [1/5, 4/5]]).log(),
        w=torch.tensor([[0.], [0.], [1.,]]),
        n=torch.tensor([2, 1, 1]),
        c_puct=torch.tensor(.001),
        seats=torch.tensor([0, 0, 0]),
        terminal=torch.tensor([False, False, False]),
        children=torch.tensor([
            [1, 2], 
            [-1, -1], 
            [-1, -1]]))

    m = cuda.mcts(**data.cuda()[None].repeat_interleave(1024, 0))
    result = cuda.descend(m)
    assert_distribution(result.parents, [0, 0, 1])
    assert_distribution(result.actions, [1/5, 4/5])

def test_balanced_cpuct():
    # Check the observed distribution satisfies the constraint
    data = arrdict.arrdict(
        logits=torch.tensor([
            [1/3, 2/3],
            [1/4, 3/4],
            [1/5, 4/5]]).log(),
        w=torch.tensor([[0.], [0.], [1.,]]),
        n=torch.tensor([2, 1, 1]),
        c_puct=torch.tensor(2.),
        seats=torch.tensor([0, 0, 0]),
        terminal=torch.tensor([False, False, False]),
        children=torch.tensor([
            [1, 2], 
            [-1, -1], 
            [-1, -1]]))

    m = cuda.mcts(**data.cuda()[None].repeat_interleave(1024, 0))
    result = cuda.descend(m)

    # Reconstruct the alpha and check it satisfies the constraint
    dist = torch.histc(result.parents, 3, 0, 2)[1:].cpu()
    p = dist/dist.sum()

    A = data.logits.shape[1]
    N = data.n[0]
    lambda_n = data.c_puct*N/(A + N)
    pi = data.logits[0].exp()
    q = (data.w[:, 0]/data.n)[data.children[0]]
    alphas = lambda_n*pi/p + q

    alpha = alphas.mean()
    unity = (lambda_n*pi/(alpha - q)).sum()

    # This is particularly imprescise at low c_puct
    assert abs(unity - 1) < .1
    
def test_terminal():
    # High cpuct, transition to node #1 is terminal
    data = arrdict.arrdict(
        logits=torch.tensor([
            [1/3, 2/3],
            [1/4, 3/4],
            [1/5, 4/5]]).log(),
        w=torch.tensor([[0.], [0.], [1.,]]),
        n=torch.tensor([2, 1, 1]),
        c_puct=torch.tensor(1000.),
        seats=torch.tensor([0, 0, 0]),
        terminal=torch.tensor([False, True, False]),
        children=torch.tensor([
            [1, 2], 
            [-1, -1], 
            [-1, -1]]))

    m = cuda.mcts(**data.cuda()[None].repeat_interleave(1024, 0))
    result = cuda.descend(m)
    assert_distribution(result.parents, [1/3, 0, 2/3])
    assert_distribution(result.actions, [1/3 + 2/3*1/5, 2/3*4/5])

def test_real():
    import pickle
    with open('output/descent/hex.pkl', 'rb') as f:
        data = pickle.load(f)
        data['c_puct'] = torch.repeat_interleave(data.c_puct[:, None], data.logits.shape[1], 1)
        data = data.cuda()

    for t in range(data.logits.shape[0]):
        m = cuda.mcts(**data[t])
        result = cuda.descend(m)

def benchmark():
    import pickle
    with open('output/descent/hex.pkl', 'rb') as f:
        data = pickle.load(f)
        data['c_puct'] = torch.repeat_interleave(data.c_puct[:, None], data.logits.shape[1], 1)
        data = data.cuda()

    results = []
    with aljpy.timer() as timer:
        torch.cuda.synchronize()
        for t in range(data.logits.shape[0]):
            m = cuda.mcts(**data[t])
            results.append(cuda.descend(m))
        torch.cuda.synchronize()
    results = arrdict.stack(results)
    time = timer.time()
    samples = results.parents.nelement()
    print(f'{1000*time:.0f}ms total, {1e9*time/samples:.0f}ns/descent')

    return results

#TODO: Test other seats, test empty children


### BACKUP TESTS

def test_backup():
    data = arrdict.arrdict(
        v=torch.tensor([[1.], [2.]]),
        w=torch.tensor([[0.], [0.]]),
        n=torch.tensor([0, 0]),
        rewards=torch.tensor([[0.], [1.]]),
        parents=torch.tensor([-1, 0]),
        terminal=torch.tensor([False, False])
    )
    bk = cuda.Backup(**data[None])

### MCTS TESTS

from .. import validation, analysis
from . import mcts, MCTSAgent

#TODO: The 'v' all need to be rewritten to test something else.
def test_trivial():
    world = validation.Win.initial(device='cuda')
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=3)

    expected = torch.tensor([[+1.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_two_player():
    world = validation.WinnerLoser.initial(device='cuda')
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=3)

    expected = torch.tensor([[+1., -1.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_depth():
    world = validation.All.initial(length=3, device='cuda')
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=15)

    expected = torch.tensor([[1/8.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_multienv():
    # Need to use a fairly complex env here to make sure we've not got 
    # any singleton dims hanging around internally. They can really ruin
    # a tester's day. 
    world = validation.All.initial(n_envs=2, length=3)
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=15)

    expected = torch.tensor([[1/8.], [1/8.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def full_game_mcts(s, n_nodes, **kwargs):
    from .. import hex
    world = hex.from_string(s, device='cuda')
    agent = validation.RandomAgent()
    return mcts(world, agent, n_nodes=n_nodes, **kwargs)

def test_planted_game():
    # black_wins = """
    # bwb
    # wbw
    # ...
    # """
    # m = full_game_mcts(black_wins, 17)

    # white_wins = """
    # wb.
    # bw.
    # wbb
    # """
    # m = full_game_mcts(white_wins, 4)

    competitive = """
    wb.
    bw.
    wb.
    """
    m = full_game_mcts(competitive, 63, c_puct=1.)
    probs = m.root().logits.exp()[0]
    assert (probs[2] > probs[8]) and (probs[5] > probs[7])

@pytest.mark.skip('Takes too long, inconclusive')
def test_full_game():
    from .. import hex
    worlds = hex.Hex.initial(128, boardsize=3, device='cuda')
    black = MCTSAgent(validation.RandomAgent(), n_nodes=32, c_puct=.5)
    white = validation.RandomAgent()
    trace = analysis.rollout(worlds, [black, white], n_reps=1)

    wins = (trace.transitions.rewards == 1).sum(0).sum(0)
    rates = wins/wins.sum()
    assert rates[0] > rates[1]

def benchmark_mcts(T=16):
    import pandas as pd
    import aljpy
    import matplotlib.pyplot as plt
    from .. import hex

    results = []
    for n in np.logspace(0, 14, 15, base=2, dtype=int):
        env = hex.Hex.initial(n_envs=n, boardsize=3, device='cuda')
        black = MCTSAgent(validation.RandomAgent(), n_nodes=16)
        white = validation.RandomAgent()

        torch.cuda.synchronize()
        with aljpy.timer() as timer:
            trace = analysis.rollout(env, [black, white], 16)
            torch.cuda.synchronize()
        results.append({'n_envs': n, 'runtime': timer.time(), 'samples': T*n})
        print(results[-1])
    df = pd.DataFrame(results)
        
    with plt.style.context('seaborn-poster'):
        ax = df.plot.scatter('n_envs', 'runtime', zorder=2)
        ax.set_xscale('log', base=2)
        ax.set_xlim(1, 2**14)
        ax.set_title('scaling of runtime w/ env count')
        ax.grid(True, zorder=1, alpha=.25)
