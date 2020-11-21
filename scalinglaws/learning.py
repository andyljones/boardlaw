import torch
import numpy as np

def batch_indices(T, B, batch_size, device='cuda'):
    batch_width = batch_size//T
    indices = torch.randperm(B, device=device)
    indices = [indices[i:i+batch_width] for i in range(0, B, batch_width)]
    return indices

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

def present_value(deltas, fallback, reset, alpha):
    # reward-to-go, reset: fall back to value
    # reward-to-go, terminal: fall back to delta
    # advantages, reset: fall back to delta
    # advantages, terminal: fall back to delta
    assert_same_shape(deltas, fallback[:-1], reset[:-1])

    result = torch.full_like(fallback, np.nan)
    result[-1] = fallback[-1]
    for t in np.arange(deltas.size(0))[::-1]:
        result[t] = torch.where(reset[t], fallback[t], deltas[t] + alpha*result[t+1])
    return result

def reward_to_go(reward, value, reset, terminal, gamma):
    # regular: final row is values, prev rows are accumulations of reward
    # next is reset: use value for current
    # next is terminal: use reward for current 
    assert (reset | ~terminal).all(), 'Some sample is marked as terminal but not reset'
    fallback = value
    fallback[terminal] = reward[terminal]
    return present_value(reward[:-1], fallback, reset, gamma).detach()

#########
# TESTS #
#########

def test_reward_to_go():
    reward = torch.tensor([1., 2., 3.])
    value = torch.tensor([4., 5., 6.])
    gamma = 1.

    reset = torch.tensor([False, False, False])
    terminal = torch.tensor([False, False, False])
    actual = reward_to_go(reward, value, reset, terminal, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([9., 8., 6.]))

    reset = torch.tensor([False, True, False])
    terminal = torch.tensor([False, False, False])
    actual = reward_to_go(reward, value, reset, terminal, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([6., 5., 6.]))

    reset = torch.tensor([False, True, False])
    terminal = torch.tensor([False, True, False])
    actual = reward_to_go(reward, value, reset, terminal, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([3., 2., 6.]))