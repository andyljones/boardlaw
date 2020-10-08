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

        randperm = torch.randperm(n_seats*n_envs, device=device, dtype=torch.long).reshape(n_envs, n_seats)
        self.agent_ids = randperm % n_agents
        self.copy_ids = randperm // n_agents

    def agentify(self, x, seats):
        agents = self.agent_ids.gather(1, seats[:, None].long()).squeeze(1)
        return [x[agents == a] for a in range(self.n_agents)]

    def envify(self, xs, seats):
        agents = self.agent_ids.gather(1, seats[:, None].long()).squeeze(1)
        env_ids = torch.arange(self.n_envs, device=agents.device)
        env_ids = torch.cat([env_ids[agents == a] for a in range(self.n_agents)])
        env_idxs = torch.argsort(env_ids)
        return arrdict.cat(xs)[env_idxs]