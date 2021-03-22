import torch
import numpy as np
from itertools import cycle
from rebar import arrdict

def mix(worlds, T=2500):
    for _ in range(T):
        actions = torch.distributions.Categorical(probs=worlds.valid.float()).sample()
        worlds, transitions = worlds.step(actions)
    return worlds

@arrdict.mapping
def half(x):
    if isinstance(x, torch.Tensor) and x.dtype == torch.float:
        return x.half()
    else:
        return x

def rel_entropy(logits):
    valid = (logits > -np.inf)
    zeros = torch.zeros_like(logits)
    logits = logits.where(valid, zeros)
    probs = logits.exp().where(valid, zeros)
    return (-(logits*probs).sum(-1).mean(), torch.log(valid.sum(-1).float()).mean())

def noise_scale(B, opt):
    step = list(opt.state.values())[0]['step']
    beta1, beta2 = opt.param_groups[0]['betas']
    m_bias = 1 - beta1**step
    v_bias = 1 - beta2**step

    m = 1/m_bias*torch.cat([s['exp_avg'].flatten() for s in opt.state.values() if 'exp_avg' in s])
    v = 1/v_bias*torch.cat([s['exp_avg_sq'].flatten() for s in opt.state.values() if 'exp_avg_sq' in s])

    # Follows from chasing the var through the defn of m
    inflator = (1 - beta1**2)/(1 - beta1)**2

    S = B*(v.mean() - m.pow(2).mean())
    G2 = inflator*m.pow(2).mean()

    return S/G2

def gather(arr, indices):
    if isinstance(arr, dict):
        return arr.__class__({k: gather(arr[k], indices[k]) for k in arr})
    return torch.gather(arr, -1, indices.type(torch.long).unsqueeze(-1)).squeeze(-1)

def flatten(arr):
    if isinstance(arr, dict):
        return torch.cat([flatten(v) for v in arr.values()], -1)
    return arr

def assert_same_shape(ref, *arrs):
    for a in arrs:
        assert ref.shape == a.shape

def present_value(deltas, fallback, terminal, alpha):
    # reward-to-go, reset: fall back to value
    # reward-to-go, terminal: fall back to delta
    # advantages, reset: fall back to delta
    # advantages, terminal: fall back to delta
    assert_same_shape(deltas, fallback[:-1], terminal[:-1])

    result = torch.full_like(fallback, np.nan)
    result[-1] = fallback[-1]
    for t in np.arange(deltas.size(0))[::-1]:
        result[t] = torch.where(terminal[t], fallback[t], deltas[t] + alpha*result[t+1])
    return result

def reward_to_go(reward, value, terminal, gamma=1.):
    # regular: final row is values, prev rows are accumulations of reward
    # next is reset: use value for current
    # next is terminal: use reward for current 
    fallback = value
    fallback[terminal] = reward[terminal]
    return present_value(reward[:-1], fallback, terminal, gamma).detach()


#########
# TESTS #
#########

def test_reward_to_go():
    reward = torch.tensor([1., 2., 3.])
    value = torch.tensor([4., 5., 6.])
    gamma = 1.

    terminal = torch.tensor([False, False, False])
    actual = reward_to_go(reward, value, terminal, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([9., 8., 6.]))

    terminal = torch.tensor([False, True, False])
    actual = reward_to_go(reward, value, terminal, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([3., 2., 6.]))
