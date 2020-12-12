import torch
import torch.cuda
from .. import cuda

loaded = cuda.load(__package__)
for k in dir(loaded):
    if not k.startswith('__'):
        globals()[k] = getattr(loaded, k)

def test():
    board = torch.zeros((1, 3, 3)).int().cuda()
    actions = torch.tensor([[0, 1]]).int().cuda()

    loaded.flood(board, actions)

    print(board[0])