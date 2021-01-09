import torch
from torch import nn
from rebar import arrdict
import pickle
from pathlib import Path
from . import common
from tqdm.auto import tqdm
from boardlaw.hex import Hex
from torch.nn import functional as F

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

class VictoryHead(nn.Module):

    def __init__(self, D):
        super().__init__()
        self.full = nn.Linear(D, 2)

    def forward(self, x, **kwargs):
        return F.log_softmax(self.full(x), -1)

def run():
    D = 32
    B = 1024
    T = 1000
    device = 'cuda'

    boards = load_boards().to(device)
    worlds = Hex(boards.worlds)

    n_boards, boardsize, _ = worlds.board.shape
    head = VictoryHead(D)
    model = common.FCModel(head, boardsize, D).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    with tqdm(total=T) as pbar:
        for t in range(T):
            idxs = torch.randint(0, n_boards, size=(B,), device=device)
            outputs = model(worlds[idxs].obs)
            targets = boards.target[idxs, 0].div(2).add(.5).int()

            loss = F.nll_loss(outputs, targets.long())

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.update(1)
            pbar.set_description(f'{loss:.2f}')
