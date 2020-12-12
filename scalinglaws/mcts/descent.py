import torch
import torch.distributions
from . import search, cuda
from rebar import arrdict
import aljpy

def assert_shape(x, s):
    assert (x.ndim == len(s)) and x.shape == s, f'Expected {s}, got {x.shape}'
    assert x.device.type == 'cuda', f'Expected CUDA tensor, got {x.device.type}'

def descend(logits, w, n, c_puct, seats, terminal, children):
    B, T, A = logits.shape
    S = w.shape[-1]
    assert_shape(w, (B, T, S))
    assert_shape(n, (B, T))
    assert_shape(c_puct, (B,))
    assert_shape(seats, (B, T))
    assert_shape(terminal, (B, T))
    assert_shape(children, (B, T, A))
    assert (c_puct > 0.).all(), 'Zero c_puct not supported; will lead to an infinite loop in the kernel'

    with torch.cuda.device(logits.device):
        result = cuda.descend(logits, w, n.int(), c_puct, seats.int(), terminal, children.int())
    return arrdict.arrdict(
        parents=result.parents, 
        actions=result.actions)

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
        c_puct=torch.tensor(.0),
        seats=torch.tensor([0]),
        terminal=torch.tensor([False]),
        children=torch.tensor([[-1, -1]]))
    
    result = descend(**data.cuda()[None].repeat_interleave(1024, 0))
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

    result = descend(**data.cuda()[None].repeat_interleave(1024, 0))

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

    result = descend(**data.cuda()[None].repeat_interleave(1024, 0))

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

    result = descend(**data.cuda()[None].repeat_interleave(8092, 0))

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

    print(unity)
    # This is particularly imprescise at low c_puct
    assert abs(unity - 1) < .05
    
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

    result = descend(**data.cuda()[None].repeat_interleave(1024, 0))

    assert_distribution(result.parents, [1/3, 0, 2/3])
    assert_distribution(result.actions, [1/3 + 2/3*1/5, 2/3*4/5])

def test_real():
    import pickle
    with open('output/descent/hex.pkl', 'rb') as f:
        data = pickle.load(f)
        data['c_puct'] = torch.repeat_interleave(data.c_puct[:, None], data.logits.shape[1], 1)
        data = data.cuda()

    for t in range(data.logits.shape[0]):
        result = descend(**data[t])

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
            results.append(descend(**data[t]))
        torch.cuda.synchronize()
    results = arrdict.stack(results)
    time = timer.time()
    samples = results.parents.nelement()
    print(f'{1000*time:.0f}ms total, {1e9*time/samples:.0f}ns/descent')

    return results

#TODO: Test other seats, test empty children