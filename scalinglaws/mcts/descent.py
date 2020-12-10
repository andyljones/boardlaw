import torch
import torch.distributions
from . import search

def action_stats(envs, children, n, w):
    mask = (children != -1)
    n = n[envs[:, None].expand_as(children), children]
    n = n.where(mask, torch.zeros_like(n))

    w = w[envs[:, None].expand_as(children), children]
    w = w.where(mask[..., None], torch.zeros_like(w))

    q = w/n[..., None]

    # Q scaling + pessimistic initialization
    q[n == 0] = 0 
    q = (q - q.min())/(q.max() - q.min() + 1e-6)
    q[n == 0] = 0 

    return q, n

def policy(logits, seats, qc, nc, c_puct):
    # all args are (envs, current)
    pi = logits.exp()

    seats = seats[:, None, None].expand(-1, qc.size(1), -1)
    q = qc.gather(2, seats.long()).squeeze(-1)

    # N == 0 leads to nans, so let's clamp it at 1
    N = nc.sum(-1).clamp(1, None)
    n_actions = q.shape[-1]
    lambda_n = c_puct*N/(n_actions + N)

    soln = search.solve_policy(pi, q, lambda_n)

    return soln.policy

def descend(logits, seats, terminal, children, n, w, c_puct):
    # What does this use?
    # * descent: envs, logits, terminal, children
    # * policy: logits, seats, c_puct, n_actions
    # * action_stats: children, n, w
    # 
    # So all together:
    # * substantial: logits, terminal, children, seats, n, w
    # * trivial: envs, n, w

    envs = torch.arange(logits.shape[0], device=logits.device)
    current = torch.full_like(envs, 0)
    actions = torch.full_like(envs, -1)
    parents = torch.full_like(envs, 0)

    while True:
        interior = ~torch.isnan(logits[envs, current]).any(-1)
        terminal = terminal[envs, current]
        active = interior & ~terminal
        if not active.any():
            break

        e, c = envs[active], current[active]

        qc, nc = action_stats(e, children[e, c], n[e, c], w[e, c])
        probs = policy(logits[e, c], seats[e, c], qc, nc, c_puct)
        sampled = torch.distributions.Categorical(probs=probs).sample()

        actions[active] = sampled
        parents[active] = c
        current[active] = children[e, c, sampled]

    return parents, actions