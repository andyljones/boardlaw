import torch
from rebar import arrdict

def symmetrize(batch):
    batch = type(batch)({k: type(v)(**v) for k, v in batch.items()})

    # Shift termination backwards a step
    terminal = batch.inputs.terminal.clone()
    terminal[:-1] = terminal[1:] | terminal[:-1]
    batch['inputs']['terminal'] = terminal

    # Mirror rewards backwards a step
    rewards = batch.responses.reward.clone()
    rewards[:-1] = rewards[:-1] - rewards[1:] 
    batch['responses']['reward'] = rewards

    return batch

def deinterlace(batch):
    seat = batch.inputs.seat
    T, B = seat.shape

    ts, bs = torch.meshgrid(
            torch.arange(T, device=seat.device),
            torch.arange(B, device=seat.device))

    ts_inv = torch.full_like(seat, -1, dtype=torch.long)
    resets = torch.full_like(seat, False, dtype=torch.bool)
    totals = ts.new_zeros(B)
    for p in [0, 1]:
        mask = batch.inputs.seat == p
        ts_inv[mask] = (totals[None, :] + mask.cumsum(0) - 1)[mask]
        
        totals += mask.sum(0)
        resets[totals-1, bs[0]] = True
    
    us = torch.full_like(ts, -1)
    us[ts_inv, bs] = ts

    deinterlaced = batch[us, bs]
    if 'reset' in deinterlaced.batch:
        resets = resets | deinterlaced.batch.reset
    deinterlaced['batch']['reset'] = resets

    return deinterlaced

class FixedMatcher:

    def __init__(self, n_agents, n_envs, n_seats, device='cuda'):
        assert n_envs*n_seats % n_agents == 0, 'The total number of players is not divisible by the number of agents'
        self.n_agents = n_agents
        self.n_envs = n_envs
        self.n_seats = n_seats
        self.n_copies = n_envs*n_seats // n_agents
        self.device = device

        randperm = torch.randperm(n_seats*n_envs, device=device, dtype=torch.long).reshape(n_envs, n_seats)
        self._agent_ids = randperm % n_agents

    def agent_ids(self, seats):
        return self._agent_ids.gather(1, seats[:, None].long()).squeeze(1)

    def agent_masks(self, seats):
        agents = self.agent_ids(seats)
        return {a: (agents == a) for a in range(self.n_agents) if (agents == a).any()}

    def agentify(self, x, seats):
        return {a: x[m] for a, m in self.agent_masks(seats).items()}

    def envify(self, xs, seats):
        env_ids = torch.arange(self.n_envs, device=self.device)
        env_ids = torch.cat([env_ids[m] for a, m in self.agent_masks(seats).items()])
        env_idxs = torch.argsort(env_ids)
        return arrdict.cat([xs[a] for a in sorted(xs)])[env_idxs]

def rollout(env, agents, n_steps=256):
    matcher = FixedMatcher(len(agents), env.n_envs, env.n_seats, device=env.device)

    trace = []
    env_inputs = env.reset()
    for _ in range(n_steps):
        agent_inputs = matcher.agentify(env_inputs, env_inputs.seat)
        decisions = {a: agents[a](ai[None], sample=True).squeeze(0) for a, ai in agent_inputs.items()}
        env_decisions = matcher.envify(decisions, env_inputs.seat)
        responses, new_inputs = env.step(env_decisions.actions)
        trace.append(arrdict.arrdict(
            agent_ids=matcher.agent_ids(env_inputs.seat),
            inputs=env_inputs,
            decisions=env_decisions,
            responses=responses))
        env_inputs = new_inputs
    
    return arrdict.stack(trace)

def winrate(env, agents):
    trace = rollout(env, agents)

    totals = torch.zeros(len(agents), device=env.device)
    totals.index_add_(0, trace.agent_ids.flatten(), trace.responses.reward.flatten())
    return totals/trace.inputs.terminal.sum()
