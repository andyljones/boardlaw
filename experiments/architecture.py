from boardlaw.main import worldfunc, agentfunc, mix, as_chunk, optimize, half
from boardlaw.learning import reward_to_go
from pavlov import storage
from rebar import arrdict
import torch
from logging import getLogger
from boardlaw import networks

log = getLogger(__name__)

def clone(sd):
    return {k: v.clone().detach() for k, v in sd.items()}

def experience(run, n_envs=8*1024, device='cuda'):
    #TODO: Restore league and sched when you go back to large boards
    worlds = mix(worldfunc(n_envs, device=device))
    agent = agentfunc(device)

    sd = storage.load_latest(run)
    agent.load_state_dict(sd['agent'])

    # Collect experience
    buffer = []
    while True:
        with torch.no_grad():
            decisions = agent(worlds, value=True)
        new_worlds, transition = worlds.step(decisions.actions)

        buffer.append(arrdict.arrdict(
            worlds=worlds,
            decisions=decisions.half(),
            transitions=half(transition)).detach())

        log.info(f'Experience: {len(buffer)}/{worlds.boardsize**2}')
        if len(buffer) > worlds.boardsize**2:
            buffer = buffer[1:]
            chunk = arrdict.stack(buffer)
            terminal = torch.stack([chunk.transitions.terminal for _ in range(chunk.worlds.n_seats)], -1)
            targets = reward_to_go(
                        chunk.transitions.rewards.float(), 
                        chunk.decisions.v.float(), 
                        terminal).half()
            
            yield chunk.worlds.obs[0], targets[0]

        worlds = new_worlds

def run():
    worlds = worldfunc(1)
    network = networks.FCModel(worlds.obs_space, worlds.action_space)
    for X, y in experience('*muddy-make'):
        break