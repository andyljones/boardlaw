import torch
import torch.cuda
import torch.testing
from .. import cuda

B = 0
W = 1

EMPTY = 0
BLACK = 1
WHITE = 2
TOP = 3
BOT = 4
LEFT = 5
RIGHT = 6

TL = 0
TC = 1
TR = 2
CL = 3
CC = 4
CR = 5
BL = 6
BC = 7
BR = 8

loaded = cuda.load(__package__)
for k in dir(loaded):
    if not k.startswith('__'):
        globals()[k] = getattr(loaded, k)

def empty_board():
    return torch.tensor([[
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]]).byte().cuda()

def tokened_board(*moves):
    board = empty_board()
    for ij, v in moves:
        i = ij // 3
        j = ij % 3
        board[:, i, j] = v
    return board

def apply(seat, action, board):
    seats = torch.tensor([seat]).int().cuda()
    actions = torch.tensor([action]).int().cuda()
    return loaded.step(board, seats, actions)

def test_move(seat, action, initial, expected):
    result = apply(seat, action, initial)
    torch.testing.assert_allclose(initial, expected)
    torch.testing.assert_allclose(result, torch.zeros_like(result))

def test_single_moves():
    # Black
    test_move(B, CC, empty_board(), tokened_board((CC, BLACK)))
    test_move(B, TL, empty_board(), tokened_board((TL, TOP)))
    test_move(B, BR, empty_board(), tokened_board((BR, BOT)))
    test_move(B, TR, empty_board(), tokened_board((TR, TOP))) # not mirrored

    # White
    test_move(W, CC, empty_board(), tokened_board((CC, WHITE)))
    test_move(W, TL, empty_board(), tokened_board((TL, LEFT)))
    test_move(W, BR, empty_board(), tokened_board((BR, RIGHT)))
    test_move(W, TR, empty_board(), tokened_board((BL, LEFT))) # mirrored

def test_wins():
    # Black win
    board = tokened_board((TC, TOP), (BC, BOT))
    result = apply(B, CC, board)
    torch.testing.assert_allclose(result, torch.tensor([+1., -1.]).cuda())

    # White win
    board = tokened_board((CL, LEFT), (CR, RIGHT))
    result = apply(W, CC, board)
    torch.testing.assert_allclose(result, torch.tensor([-1., +1.]).cuda())

def test_flooding():
    # Bottom flooding
    initial = tokened_board((CL, BLACK), (CC, BLACK))
    expected = tokened_board((CL, BOT), (CC, BOT), (BC, BOT))
    test_move(B, BC, initial, expected)

    # Left flooding
    initial = tokened_board((TC, WHITE), (CC, WHITE))
    expected = tokened_board((TC, LEFT), (CC, LEFT), (CL, LEFT))
    test_move(W, TC, initial, expected)
