import torch

def deinterlace(batch):
    player = batch.inputs.player
    T, B = player.shape

    ts, bs = torch.meshgrid(
            torch.arange(T, device=player.device),
            torch.arange(B, device=player.device))

    ts_inv = torch.full_like(player, -1, dtype=torch.long)
    resets = torch.full_like(player, False, dtype=torch.bool)
    totals = ts.new_zeros(B)
    for p in [0, 1]:
        mask = batch.inputs.player == p
        ts_inv[mask] = (totals[None, :] + mask.cumsum(0) - 1)[mask]
        
        totals += mask.sum(0)
        resets[totals-1, bs[0]] = True
    
    us = torch.full_like(ts, -1)
    us[ts_inv, bs] = ts

    deinterlaced = batch[us, bs]
    assert 'reset' not in deinterlaced.inputs

    #TODO: This should be done seperately in a prior step.

    # Shift termination backwards a step
    terminal = batch.inputs.terminal.clone()
    terminal[:-1] = terminal[1:] | terminal[:-1]
    deinterlaced['inputs']['reset'] = resets | terminal[us, bs]

    # Mirror rewards backwards a step
    rewards = batch.responses.reward.clone()
    rewards[:-1] = rewards[:-1] - rewards[1:] 
    deinterlaced['responses']['reward'] = rewards

    return deinterlaced