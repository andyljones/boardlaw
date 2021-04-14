import torch
from .arena import common
from boardlaw import learning, main
from rebar import arrdict

def collect(run, idx, n_envs=32*1024):
    agent = common.agent(run, idx, 'cuda')
    worlds = common.worlds(run, n_envs, 'cuda')

    buffer = []
    while True:
        while len(buffer) < 64:
            with torch.no_grad():
                decisions = agent(worlds, value=True)
            new_worlds, transition = worlds.step(decisions.actions)

            buffer.append(arrdict.arrdict(
                worlds=worlds,
                decisions=decisions.half(),
                transitions=learning.half(transition)).detach())

            worlds = new_worlds

        chunk, buffer = main.as_chunk(buffer, n_envs)
        
        mixness = chunk.transitions.terminal.float().mean(1)
        mixness = (mixness.max() - mixness.min())/mixness.median()
        if mixness < .25:
            break

    return chunk