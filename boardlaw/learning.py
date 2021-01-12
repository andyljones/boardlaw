import torch
import numpy as np
from itertools import cycle

PRIME = 160481219

def batch_indices_fancy(buffer_length, buffer_inc, n_envs, batch_size, device='cuda'):
    #TODO: This is supposed to account for the movement of the buffer so that each sample gets used exactly
    # once. But it doesn't work. 
    assert n_envs <= batch_size
    assert batch_size % n_envs == 0 
    assert (n_envs*buffer_length) % batch_size == 0

    batch_height = batch_size // n_envs
    cols = torch.arange(n_envs, device=device)[None, :].repeat_interleave(batch_height, 1)

    offsets = PRIME*torch.arange(n_envs, device=device)

    rowperm = torch.arange(buffer_length, device=device)
    i = 0
    while True:
        start = batch_height*i % buffer_length
        end = start + batch_height

        seeds = rowperm[start:end] - i*buffer_inc
        rows = (seeds[:, None] + offsets[None, :]) % buffer_length

        yield (rows, cols)

        i += 1

def batch_indices(buffer_length, n_envs, batch_size, device='cuda'):
    while True:
        cols = torch.arange(batch_size, device=device) % n_envs
        rows = torch.randperm(batch_size) % buffer_length
        yield rows, cols

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

def test_batch_indices():
    buffer_length = 16
    buffer_inc = 4
    n_envs = 1

    batch_size = 4

    n_incs = 16
    counts = torch.zeros(buffer_length+n_incs*buffer_inc)
    buffer = torch.arange(buffer_length)

    idxs = batch_indices_fancy(buffer_length, buffer_inc, n_envs, batch_size, device='cpu')
    for _, (rows, cols) in zip(range(n_incs), idxs):
        counts[buffer[rows]] += 1

        buffer = torch.cat([
            buffer[buffer_inc:],
            torch.arange(buffer.max()+1, buffer.max()+1+buffer_inc)])

