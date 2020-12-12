import numpy as np
from . import Hex, _CHARS
import torch
import torch.distributions
import torch.cuda

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
    strs = _CHARS
    board = env.board[e].clone()
    strings = np.vectorize(strs.__getitem__)(board.cpu().numpy())
    return '\n'.join(' '*i + ' '.join(r) for i, r in enumerate(strings))

def test_open_spiel():
    import pyspiel

    e = 1
    ours = Hex.initial(3, 11, device='cpu')

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

def benchmark(n_envs=1024, n_steps=512):
    import aljpy
    worlds = Hex.initial(n_envs)

    torch.cuda.synchronize()
    with aljpy.timer() as timer:
        for _ in range(n_steps):
            actions = torch.distributions.Categorical(probs=worlds.valid.float()).sample()
            worlds, _ = worlds.step(actions)
        torch.cuda.synchronize()
    print(f'{n_envs*n_steps/timer.time():.0f} samples/sec')