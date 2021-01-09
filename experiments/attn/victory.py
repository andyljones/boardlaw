import torch
from torch import nn
from rebar import arrdict
import pickle
from pathlib import Path
from . import common
from tqdm.auto import tqdm
from boardlaw.hex import Hex
from torch.nn import functional as F

def gamegen(n_envs=1024):
    from boardlaw.main import worldfunc, agentfunc
    from pavlov import storage

    worlds = worldfunc(n_envs)
    agent = agentfunc()
    agent.evaluator = agent.evaluator.prime

    sd = storage.load_snapshot('*kind-june*', 2)
    agent.load_state_dict(sd['agent'])

    while True:
        with torch.no_grad():
            decisions = agent(worlds, eval=False)
        new_worlds, transitions = worlds.step(decisions.actions)

        if transitions.terminal.any():
            yield arrdict.arrdict(
                worlds=arrdict.to_dicts(worlds[transitions.terminal]),
                actions=decisions.actions[transitions.terminal],
                rewards=transitions.rewards[transitions.terminal])

        worlds = new_worlds

        
def save_boards(total=32*1024):
    boards = []
    with tqdm(total=total) as pbar:
        for b in gamegen(total):
            boards.append(b)
            count = sum(b.actions.size(0) for b in boards)
            pbar.update(count - pbar.n)
            if count >= total:
                break

    boards = arrdict.cat(boards)
    Path('output/terminal-boards.pkl').write_bytes(pickle.dumps(boards))

def load_boards():
    return pickle.loads(Path('output/terminal-boards.pkl').read_bytes())

def run():
    D = 32
    B = 8*1024
    T = 10000
    device = 'cuda'

    boards = arrdict.from_dicts(load_boards()).to(device)
    worlds = Hex(boards.worlds)

    n_boards, boardsize, _ = worlds.board.shape
    model = common.AttnModel(common.PosActions, boardsize, D).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    with tqdm(total=T) as pbar:
        for t in range(T):
            idxs = torch.randint(0, n_boards, size=(B,), device=device)
            outputs = model(worlds[idxs].obs)

            loss = F.nll_loss(outputs.reshape(B, -1), boards.actions[idxs])

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.update(1)
            pbar.set_description(f'{loss:.2f}')
