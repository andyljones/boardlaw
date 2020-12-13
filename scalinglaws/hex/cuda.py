import torch
import torch.cuda
import torch.testing
from .. import cuda

loaded = cuda.load(__package__)
for k in dir(loaded):
    if not k.startswith('__'):
        globals()[k] = getattr(loaded, k)

def empty_board():
    return torch.tensor([[
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]]).int().cuda()

def move_board(*moves):
    board = empty_board()
    for i, j, v in moves:
        board[:, i, j] = v
    return board

def test_move(seat, action, initial, expected):
    seats = torch.tensor([seat]).int().cuda()
    actions = torch.tensor([action]).int().cuda()
    
    result = loaded.step(initial, seats, actions)
    torch.testing.assert_allclose(initial, expected)
    return result

def test_single_moves():
    # Black
    test_move(0, 4, empty_board(), move_board((1, 1, 1))) # black
    test_move(0, 0, empty_board(), move_board((0, 0, 3))) # top 
    test_move(0, 8, empty_board(), move_board((2, 2, 4))) # bot

    # White
    test_move(1, 4, empty_board(), move_board((1, 1, 5))) # white
    test_move(1, 0, empty_board(), move_board((0, 0, 7))) # left
    test_move(1, 8, empty_board(), move_board((2, 2, 8))) # right

def test_wins():
    # Black win
    result = test_move(0, 4, 
        move_board((0, 1, 3), (2, 1, 4)),
        move_board((0, 1, 3), (2, 1, 4), (1, 1, 2)))
    torch.testing.assert_allclose(result, torch.tensor([1., 0.]).cuda())

    # White win
    result = test_move(1, 4, 
        move_board((1, 0, 7), (1, 2, 8)),
        move_board((1, 0, 7), (1, 2, 8), (1, 1, 6)))
    torch.testing.assert_allclose(result, torch.tensor([0., 1.]).cuda())