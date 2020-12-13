import torch
import torch.cuda
from .. import cuda

loaded = cuda.load(__package__)
for k in dir(loaded):
    if not k.startswith('__'):
        globals()[k] = getattr(loaded, k)

def test():
    board = torch.zeros((1, 3, 3)).int().cuda()
    seats = torch.zeros((1,)).int().cuda()
    actions = torch.tensor([1]).int().cuda()

    results = loaded.step(board, seats, actions)

    print(board[0])