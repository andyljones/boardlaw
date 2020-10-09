import torch
from . import matchers
from rebar import arrdict, recording, storing
import numpy as np
from tqdm import tqdm

def rollout(env, agents):
    matcher = matchers.FixedMatcher(len(agents), env.n_envs, env.n_seats, device=env.device)

    env_inputs = env.reset()
    while True:
        agent_inputs = matcher.agentify(env_inputs, env_inputs.seat)
        decisions = {a: agents[a](ai[None], sample=True).squeeze(0) for a, ai in agent_inputs.items()}
        env_decisions = matcher.envify(decisions, env_inputs.seat)
        state = env.state()
        responses, new_inputs = env.step(env_decisions.actions)
        yield arrdict.arrdict(
                    agent_ids=matcher.agent_ids(env_inputs.seat),
                    inputs=env_inputs,
                    decisions=env_decisions,
                    responses=responses,
                    state=state)
        env_inputs = new_inputs

def trace(env, agents, n_trajs=None):
    n_trajs = env.n_trajs if n_trajs is None else n_trajs

    trace = []
    trajs = 0
    for t in rollout(env, agents):
        trajs += t.inputs.terminal.sum()
        if trajs >= n_trajs:
            break
        trace.append(t)
    
    return arrdict.stack(trace)

def watch(envfunc, agents, N=None, fps=3):
    env = envfunc(1)
    assert len(agents) == env.n_seats

    with recording.ParallelEncoder(env.plot_state, N=N, fps=fps) as encoder:
        for t in rollout(env, agents):
            if t.inputs.terminal.any():
                break
            encoder(arrdict.numpyify(t.state))
    
    encoder.notebook()
    return encoder

def winrate(env, agents, ci=.05):
    # Assume fair coin, normal approx 
    n_trajs = (1/4*1.96)/ci**2
    trace = rollout(env, agents, n_trajs=n_trajs)

    totals = torch.zeros(len(agents), device=env.device)
    totals.index_add_(0, trace.agent_ids.flatten(), trace.responses.reward.flatten())
    return (totals/totals.sum()).cpu().numpy()

def compare(envfunc, agentfunc, run_name=-1, left=-2, right=-1):
    env = envfunc(256)
    agent = [agentfunc(env.obs_space, env.action_space).to(env.device) for _ in range(2)]
    agent[0].load_state_dict(storing.load_periodic(run_name, idx=left)['agent'])
    agent[1].load_state_dict(storing.load_periodic(run_name, idx=right)['agent'])
    return matchers.winrate(env, agent)

def compare_all(run_name=-1, n=None):
    import pandas as pd
    from itertools import combinations

    df = storing.stored_periodic(run_name)
    if (n is not None) and (n < len(df)):
        df = df.sample(n).sort_index()
    rates = {(l, r): compare(-1, l, r) for l, r in combinations(df.index, 2)}
    rates = pd.DataFrame.from_dict(rates).iloc[1].unstack(1)
    
    return rates

def ratings(**kwargs):
    import cvxpy as cp
    import pandas as pd

    rates = compare_all(**kwargs)

    names = list(set(rates.index) | set(rates.columns))
    full = rates.reindex(index=names, columns=names)
    comp = full.combine_first(1 - full.T).fillna(.5)

    targets = np.log2(1/comp.values - 1).clip(-10, +10)

    #TODO: Is this even remotely how to calculate Elos?
    r = cp.Variable(len(names))
    loss = cp.sum_squares(r[:, None] - r[None, :] - targets)
    cp.Problem(cp.Minimize(loss), [r[0] == 0]).solve()
    return pd.Series(r.value, names)