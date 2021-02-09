import time
import numpy as np
from . import Hex, cuda, CHARS
import torch
import torch.distributions
import torch.cuda
import torch.testing

OPEN_SPIEL_CHARS = 'bw'

### CUDA

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
    return cuda.step(board, seats, actions)

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

def strip(s):
    return '\n'.join(l.strip() for l in s.splitlines() if l.strip())

def board_size(s):
    return len(strip(s).splitlines())

def board_actions(s):
    size = board_size(s)
    board = (np.frombuffer((strip(s) + '\n').encode(), dtype='S1')
                 .reshape(size, size+1)
                 [:, :-1])
    indices = np.indices(board.shape)

    bs = indices[:, board == b'b'].T
    ws = indices[:, board == b'w'].T

    assert len(bs) - len(ws) in {0, 1}

    actions = []
    for i in range(len(ws)):
        actions.append([bs[i, 0], bs[i, 1]])
        actions.append([ws[i, 1], ws[i, 0]])

    if len(ws) < len(bs):
        actions.append([bs[-1, 0], bs[-1, 1]])

    return torch.tensor(actions)

def from_string(s, **kwargs):
    """Example:
    
    s = '''
    bwb
    wbw
    ...
    '''
    
    """
    worlds = Hex.initial(n_envs=1, boardsize=board_size(s), **kwargs)
    for a in board_actions(s).to(worlds.device):
        worlds, trans = worlds.step(a[None])
    return worlds

def test_basic():
    worlds = Hex.initial(1, 3, device='cpu')

    for _ in range(20):
        actions = torch.distributions.Categorical(probs=worlds.valid.float()).sample()
        worlds, _ = worlds.step(actions)

def open_spiel_board(state):
    # state ordering taken from hex.h 
    strs = 'W<>w.bv^B'
    board = np.array(state.observation_tensor()).reshape(9, 11, 11).argmax(0)
    strs = np.vectorize(strs.__getitem__)(board)
    return '\n'.join(' '*i + ' '.join(r) for i, r in enumerate(strs))

def open_spiel_display_str(env, e):
    board = env.board[e].clone()
    strings = np.vectorize(CHARS.__getitem__)(board.cpu().numpy())
    return '\n'.join(' '*i + ' '.join(r) for i, r in enumerate(strings))

def test_open_spiel():
    import pyspiel

    e = 1
    ours = Hex.initial(64, 11, device='cpu')

    theirs = pyspiel.load_game("hex")
    state = theirs.new_initial_state()
    while True:
        seat = ours.seats[e]
        our_action = torch.distributions.Categorical(probs=ours.valid.float()).sample()
        ours, transitions = ours.step(our_action)

        if seat == 0:
            their_action = our_action[e]
        else: #if new.player == 1:
            r, c = our_action[e]//ours.boardsize, our_action[e] % ours.boardsize
            their_action = c*ours.boardsize + r

        state.apply_action(their_action)
            
        if transitions.terminal[e]:
            assert state.is_terminal()
            break
            
        our_state = open_spiel_display_str(ours, e)
        their_state = open_spiel_board(state)
        assert our_state == their_state

def benchmark_step(n_envs=4096, n_steps=1024):
    worlds = Hex.initial(n_envs)

    for _ in range(n_steps):
        actions = torch.distributions.Categorical(probs=worlds.valid.float()).sample()
        worlds, transitions = worlds.step(actions)

    actions = torch.distributions.Categorical(probs=worlds.valid.float()).sample().int()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_steps):
        cuda.step(worlds.board, worlds.seats, actions)
    torch.cuda.synchronize()
    print(f'{n_envs*n_steps/(time.time() - start):.0f} samples/sec')

def benchmark_obs(n_envs=4096, n_steps=1024):
    worlds = Hex.initial(n_envs, boardsize=9)
    worlds['seats'] = torch.arange(worlds.n_envs, device=worlds.device) % 2

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_steps):
        cuda.observe(worlds.board, worlds.seats)
    torch.cuda.synchronize()
    print(f'{n_envs*n_steps/(time.time() - start):.0f} samples/sec')

def obs_test():
    board = torch.tensor([[[0, 0], [0, 0]]]).type(torch.uint8).cuda()
    seats = torch.tensor([1,]).type(torch.long).cuda()
    cuda.observe(board, seats)

def overwrite_board():
    # Black to play on (8, 4); replaces white :/
    bad = """
        .....w.....
        ...b..w....
        ....b..w...
        .....b.....
        .ww...b....
        .bw........
        .bw..b.....
        .bwb.bw....
        .bw.ww.b...
        .bw........
        ..........."""

    from_string(bad).display()