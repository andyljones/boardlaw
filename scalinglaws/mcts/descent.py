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
    # E: num envs, T: num nodes, A: num actions, S: num seats
    # logits: (E, T, A)
    # seats: (E,)
    # terminal: (E,)
    # children: (E, T, A)
    # n: (E, T)
    # w: (E, T, S)

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

def descend_single(logits, seats, terminal, children, n, w, c_puct):
    # T: num nodes, A: num actions, S: num seats
    # logits: (T, A)
    # seats: (T,)
    # terminal: ()
    # children: (T, A)
    # n: (T)
    # w: (T, S)

    A = logits.shape[1]
    q = w/(n[:, None] + 1e-6)
    qmin, qmax = q.min(), q.max()

    t = 0
    parent, action = 0, -1
    while True:
        exterior = torch.isnan(logits[t]).any()
        if exterior or terminal[t]:
            return parent, action

        N = 0
        seat = seats[t]
        q = torch.zeros((A,), device=w.device)
        pi = logits[t].exp()
        for i in range(A):
            child = children[t, i]

            if (child > -1):
                qchild = w[child, seat]/(n[child] + 1e-6)
                qchild = (qchild - qmin)/(qmax - qmin + 1e-6)
                q[i] = qchild

                N += n[child]
            else:
                N += 1

        lambda_n = c_puct*N/(A + N)

        probs = search.solve_policy(pi[None], q[None], lambda_n[None]).policy[0]
        action = torch.distributions.Categorical(probs=probs).sample()

        parent = t
        t = children[t, action]

def test():
    import pickle
    with open('output/descent/benchmark.pkl', 'rb') as f:
        data = pickle.load(f)
        data['c_puct'] = torch.repeat_interleave(data.c_puct[:, None], data.logits.shape[1], 1)

    d = data[30, 0]
    descend_single(**d)

    
