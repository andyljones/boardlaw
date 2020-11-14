import torch
import numpy as np
from rebar import arrdict
from . import heads
import matplotlib as mpl
import matplotlib.pyplot as plt

def as_mask(m, n_envs, device):
    if not isinstance(m, torch.Tensor):
        dtype = {bool: torch.bool, int: torch.long}[type(m[0])]
        m = torch.as_tensor(m, dtype=dtype)
    if m.device != device:
        m = m.to(device)
    if m.dtype == torch.long:
        mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
        mask[m] = True
        m = mask
    assert isinstance(m, torch.Tensor) and m.dtype == torch.bool
    return m



class Hex:
    """Based on `OpenSpiel's implementation <https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/hex.cc>`_.
    """

    # Empty, 
    # black, black win, black-north-connected, black-south-connected
    # white, white win, white-west-connected, white-east-connected
    _STRINGS = '.bB^vwW<>'

    def __init__(self, n_envs=1, boardsize=11, device='cuda'):
        self.n_envs = n_envs
        self.n_seats = 2
        self.boardsize = boardsize
        self.device = torch.device(device)

        self._STATES = {s: torch.tensor(i, dtype=torch.int, device=device) for i, s in enumerate(self._STRINGS)}

        self._IS_EDGE = {
            '^': lambda idxs: idxs[..., 0] == 0,
            'v': lambda idxs: idxs[..., 0] == boardsize-1,
            '<': lambda idxs: idxs[..., 1] == 0,
            '>': lambda idxs: idxs[..., 1] == boardsize-1}

        self._NEIGHBOURS = torch.tensor([(-1, 0), (-1, +1), (0, -1), (0, +1), (+1, -1), (+1, +0)], device=device, dtype=torch.long)

        self._board = torch.full((n_envs, boardsize, boardsize), 0, device=device, dtype=torch.int)

        # As per OpenSpiel and convention, black plays first.
        self._seat = torch.full((n_envs,), 0, device=device, dtype=torch.int)
        self._envs = torch.arange(self.n_envs, device=device)
        self._step = torch.zeros(self.n_envs, device=device)

        self.obs_space = heads.Tensor((self.boardsize, self.boardsize, 2))
        self.action_space = heads.Masked(self.boardsize*self.boardsize)

    def _states(self, idxs, val=None):
        if idxs.size(-1) == 2:
            rows, cols = idxs[..., 0], idxs[..., 1]
            envs = self._envs[(slice(None),) + (None,)*(idxs.ndim-2)].expand_as(rows)
        else: # idxs.size(-1) == 3
            envs, rows, cols = idxs[..., 0], idxs[..., 1], idxs[..., 2]
        
        if val is None:
            return self._board[envs, rows, cols]
        else:
            self._board[envs, rows, cols] = val
    
    def _terminate(self, terminate):
        self._board[terminate] = self._STATES['.']
        self._seat[terminate] = 0

    def _neighbours(self, idxs):
        if idxs.size(1) == 3:
            neighbours = self._neighbours(idxs[:, 1:])
            envs = idxs[:, None, [0]].expand(-1, len(self._NEIGHBOURS), 1)
            return torch.cat([envs, neighbours], 2)
        return (idxs[:, None, :] + self._NEIGHBOURS).clamp(0, self.boardsize-1)

    def _colours(self, x):
        colours = x.clone()
        colours[(x == self._STATES['^']) | (x == self._STATES['v'])] = self._STATES['b']
        colours[(x == self._STATES['<']) | (x == self._STATES['>'])] = self._STATES['w']
        return colours

    def _flood(self, actions):
        # This eats 70% of the game's runtime.
        moves = self._states(actions)
        colors = self._colours(moves)

        active = torch.stack([moves == self._STATES[s] for s in '<>^v'], 0).any(0)

        idxs = torch.cat([self._envs[:, None], actions], 1)[active]
        while idxs.size(0) > 0:
            self._states(idxs, moves[idxs[:, 0]])
            neighbour_idxs = self._neighbours(idxs)
            possible = self._states(neighbour_idxs) == colors[idxs[:, 0], None]

            touched = torch.zeros_like(self._board, dtype=torch.bool)
            touched[tuple(neighbour_idxs[possible].T)] = True
            idxs = touched.nonzero()

    def _update_states(self, actions):
        if actions.ndim == 1:
            actions = torch.stack([actions // self.boardsize, actions % self.boardsize], -1)

        # White player sees a transposed board, so their actions need transposing back.
        black_actions = actions
        white_actions = actions.flip(1)
        actions = torch.where(self._seat[:, None] == 0, black_actions, white_actions)

        assert (self._states(actions) == 0).all(), 'One of the actions is to place a token on an already-occupied cell'

        neighbours = self._states(self._neighbours(actions))

        black = self._seat == 0
        white = self._seat == 1
        conns = {s: ((neighbours == self._STATES[s]).any(-1)) | self._IS_EDGE[s](actions) for s in self._IS_EDGE}

        new_state = torch.zeros_like(self._states(actions))
        
        new_state[black] = self._STATES['b']
        new_state[black & conns['^']] = self._STATES['^']
        new_state[black & conns['v']] = self._STATES['v']
        new_state[black & conns['^'] & conns['v']] = self._STATES['B']

        new_state[white] = self._STATES['w']
        new_state[white & conns['<']] = self._STATES['<']
        new_state[white & conns['>']] = self._STATES['>']
        new_state[white & conns['<'] & conns['>']] = self._STATES['W']

        self._states(actions, new_state)
        self._flood(actions)

        terminal = ((new_state == self._STATES['B']) | (new_state == self._STATES['W']))

        return terminal

    def _observe(self):
        black_view = torch.stack([
            torch.stack([self._board == self._STATES[s] for s in 'b^vB']).any(0),
            torch.stack([self._board == self._STATES[s] for s in 'w<>W']).any(0)], -1).float()

        # White player sees a transposed board
        white_view = black_view.transpose(1, 2).flip(3)
        obs = black_view.where(self._seat[:, None, None, None] == 0, white_view)

        return arrdict.arrdict(
            obs=obs,
            valid=(obs == 0).all(-1).reshape(self.n_envs, -1),
            seats=self._seat,
            step=self._step).clone()

    def reset(self):
        terminal = torch.ones(self.n_envs, dtype=bool, device=self.device)
        self._terminate(terminal)
        return self._observe()

    def step(self, actions):
        """Args:
            actions: (n_env, 2)-int tensor between (0, 0) and (boardsize, boardsize). Cells are indexed in row-major
            order from the top-left.
            
        Returns:

        """
        terminal = self._update_states(actions)
        rewards = torch.zeros((self.n_envs, self.n_seats), device=self.device)
        rewards[self._envs, self._seat.long()] = terminal.float()
        rewards[self._envs, 1-self._seat.long()] = -terminal.float()

        self._seat = 1 - self._seat
        self._step += 1
        self._terminate(terminal)
        responses = arrdict.arrdict(
            terminal=terminal, 
            rewards=rewards)
        return responses, self._observe()

    def state_dict(self):
        return arrdict.arrdict(
            board=self._board, 
            step=self._step,
            seat=self._seat).clone()

    def load_state_dict(self, sd):
        self._board[:] = sd.board
        self._seat[:] = sd.seat
        self._step[:] = sd.step

    def __getitem__(self, m):
        m = as_mask(m, self.n_envs, self.device)
        n_envs = m.sum()
        subenv = type(self)(n_envs, self.boardsize, self.device)
        substate = self.state_dict()[m]
        subenv.load_state_dict(substate)
        return subenv

    def __setitem__(self, m, subenv):
        m = as_mask(m, self.n_envs, self.device)
        current = self.state_dict()
        current[m] = subenv.state_dict()
        self.load_state_dict(current)

    @classmethod
    def plot_state(cls, state, e=0, ax=None):
        board = state[e].board
        width = board.shape[1]

        ax = plt.subplots()[1] if ax is None else ax
        ax.set_aspect(1)

        sin60 = np.sin(np.pi/3)
        ax.set_xlim(-1.5, 1.5*width)
        ax.set_ylim(-sin60, sin60*width)

        rows, cols = np.indices(board.shape)
        coords = np.stack([
            cols + .5*np.arange(board.shape[0])[:, None],
            # Hex centers are 1 apart, so distances between rows are sin(60)
            sin60*(board.shape[0] - 1 - rows)], -1).reshape(-1, 2)

        black = 'dimgray'
        white = 'lightgray'
        colors = ['tan'] + [black]*4 + [white]*4
        colors = np.vectorize(colors.__getitem__)(board).flatten()


        tl, tr = (-1.5, (width)*sin60), (width-.5, (width)*sin60)
        bl, br = (width/2-1, -sin60), (1.5*width, -sin60)
        ax.add_patch(mpl.patches.Polygon(np.array([tl, tr, bl, br]), linewidth=1, edgecolor='k', facecolor=black, zorder=1))
        ax.add_patch(mpl.patches.Polygon(np.array([tl, bl, tr, br]), linewidth=1, edgecolor='k', facecolor=white, zorder=1))

        radius = .5/sin60
        data_to_pixels = ax.transData.get_matrix()[0, 0]
        pixels_to_points = 1/ax.figure.get_dpi()*72.
        size = np.pi*(data_to_pixels*pixels_to_points*radius)**2
        sizes = (size,)*len(coords)

        hexes = mpl.collections.RegularPolyCollection(
                        numsides=6, 
                        sizes=sizes,
                        offsets=coords, 
                        facecolors=colors, 
                        edgecolor='k', 
                        linewidths=1, 
                        transOffset=ax.transData,
                        zorder=2)

        ax.add_collection(hexes)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

        return ax.figure

    def display(self, e=0):
        ax = self.plot_state(arrdict.numpyify(self.state_dict()), e=e)
        plt.close(ax.figure)
        return ax

## TESTS ##

def board_size(s):
    return len(s.strip().splitlines())

def board_actions(s):
    size = board_size(s)
    board = (np.frombuffer((s.strip() + '\n').encode(), dtype='S1')
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
    env = Hex(boardsize=board_size(s), **kwargs)
    for a in board_actions(s):
        response, inputs = env.step(a[None])
    return env, inputs

def test_basic():
    h = Hex(1, 3, device='cpu')

    n = h.reset()

    for _ in range(20):
        mask = (~n.obs.any(-1))

        options = torch.nonzero(mask)
        move = options[torch.randint(options.size(0), ()), 1:]
        
        o, n = h.step(move[None])

def open_spiel_board(state):
    # state ordering taken from hex.h 
    strs = 'W<>w.bv^B'
    board = np.array(state.observation_tensor()).reshape(9, 11, 11).argmax(0)
    strs = np.vectorize(strs.__getitem__)(board)
    return '\n'.join(' '*i + ' '.join(r) for i, r in enumerate(strs))

def open_spiel_display_str(env, e):
    strs = env._STRINGS
    board = env._board[e].clone()
    strings = np.vectorize(strs.__getitem__)(board.cpu().numpy())
    return '\n'.join(' '*i + ' '.join(r) for i, r in enumerate(strings))

def test_open_spiel():
    import pyspiel

    e = 1
    ours = Hex(3, 11, device='cpu')
    new = ours.reset()

    theirs = pyspiel.load_game("hex")
    state = theirs.new_initial_state()
    while True:
        seat = new.seat[e]
        our_action = torch.distributions.Categorical(probs=new.mask.float()).sample()
        old, new = ours.step(our_action)

        if seat == 0:
            their_action = our_action[e]
        else: #if new.player == 1:
            r, c = our_action[e]//ours.boardsize, our_action[e] % ours.boardsize
            their_action = c*ours.boardsize + r

        state.apply_action(their_action)
            
        if new.terminal[e]:
            assert state.is_terminal()
            break
            
        our_state = open_spiel_display_str(ours, e)
        their_state = open_spiel_board(state)
        assert our_state == their_state

def benchmark(n_envs=4096, n_steps=256):
    import aljpy
    env = Hex(n_envs)

    inputs = env.reset()
    torch.cuda.synchronize()
    with aljpy.timer() as timer:
        for _ in range(n_steps):
            actions = torch.distributions.Categorical(probs=inputs.valid.float()).sample()
            _, inputs = env.step(actions)
        
        torch.cuda.synchronize()
    print(f'{n_envs*n_steps/timer.time():.0f} samples/sec')

def test_subenvs():
    env = hex.Hex(n_envs=3, boardsize=5, device='cpu')
    inputs = env.reset()
    subenv = env[[1]]
    subresponse, subinputs = subenv.step(torch.tensor([[0, 0]], dtype=torch.long))
    env[[1]] = subenv

    board = env.state_dict().board
    assert (board[[0, 2]] == 0).all()
    assert (board[1][1:, :] == 0).all()
    assert (board[1][:, 1:] == 0).all()
    assert (board[1][0, 0] != 0).all()