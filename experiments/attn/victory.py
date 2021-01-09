import torch
from rebar import arrdict
import pickle
from pathlib import Path
from . import common

def gamegen(batchsize=1024):
    from boardlaw.main import worldfunc, agentfunc
    from pavlov import storage

    n_envs = batchsize
    worlds = worldfunc(n_envs)
    agent = agentfunc()
    agent.evaluator = agent.evaluator.prime

    sd = storage.load_snapshot('*kind-june*', 2)
    agent.load_state_dict(sd['agent'])

    buffer = []
    while True:
        with torch.no_grad():
            decisions = agent(worlds, eval=False)
        new_worlds, transitions = worlds.step(decisions.actions)

        if transitions.terminal.any():
            buffer.append(arrdict.arrdict(
                worlds=worlds[transitions.terminal],
                target=transitions.rewards[transitions.terminal]))

        worlds = new_worlds

        size = sum(b.target.size(0) for b in buffer)
        if size > batchsize:
            buffer = arrdict.cat(buffer)
            yield buffer[:batchsize]
            buffer = [buffer[batchsize:]]
        
def save_boards():
    boards = []
    for b in gamegen(8192):
        boards.append(b)
        print(len(boards))
        if len(boards) > 64:
            break

    boards = arrdict.cat(boards)
    Path('output/terminal-boards.pkl').write_bytes(pickle.dumps(boards))

def load_boards():
    return pickle.loads(Path('output/terminal-boards.pkl').read_bytes())

def victory_test():
    boards = load_boards()
    boardsize = boards.size(-1)

    D = 32
    model = common.Model(boardsize).cuda()

