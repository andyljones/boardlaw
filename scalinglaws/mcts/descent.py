import torch
import torch.distributions
from . import search, cuda
from rebar import arrdict
import aljpy

def descend_cuda(logits, w, n, c_puct, seats, terminal, children):
    result = cuda.descend(logits, w, n.int(), c_puct, seats.int(), terminal, children.int())
    return arrdict.arrdict(
        parents=result.parents, 
        actions=result.actions)

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
            results.append(descend_cuda(**data[t]))
        torch.cuda.synchronize()
    results = arrdict.stack(results)
    time = timer.time()
    samples = results.parents.nelement()
    print(f'{1000*time:.0f}ms total, {1e9*time/samples:.0f}ns/descent')

    return results

def test():
    import pickle
    with open('output/descent/hex.pkl', 'rb') as f:
        data = pickle.load(f)
        data['c_puct'] = torch.repeat_interleave(data.c_puct[:, None], data.logits.shape[1], 1)
        data = data.cuda()

    torch.manual_seed(2)
    result = descend_cuda(**data[-1,:3])
    print(result.parents, result.actions)
