import torch
import numpy as np
from rebar import arrdict
from . import heads
import matplotlib as mpl
import matplotlib.pyplot as plt

def unique(idxs, maxes):
    if idxs.size(0) == 0:
        return idxs

    assert idxs.size(-1) == len(maxes)+1
    maxes = [idxs[..., 0].max()+1] + maxes

    base = 1
    id = torch.zeros_like(idxs[..., 0])
    for d in reversed(range(len(maxes))):
        id += idxs[..., d]*base
        base *= maxes[d]
    
    uid = torch.unique(id)

    unique_idxs = []
    for d in reversed(range(len(maxes))):
        uidx = uid % maxes[d]
        uid = (uid - uidx)//maxes[d]
        unique_idxs.append(uidx)

    return torch.stack(unique_idxs[::-1], -1)

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
        moves = self._states(actions)
        colors = self._colours(moves)

        active = torch.stack([moves == self._STATES[s] for s in '<>^v'], 0).any(0)

        idxs = torch.cat([self._envs[:, None], actions], 1)[active]
        while idxs.size(0) > 0:
            self._states(idxs, moves[idxs[:, 0]])
            neighbour_idxs = self._neighbours(idxs)
            possible = self._states(neighbour_idxs) == colors[idxs[:, 0], None]
            #TODO: Take uniques
            idxs = unique(neighbour_idxs[possible], [self.boardsize, self.boardsize])

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
            mask=(obs == 0).all(-1).reshape(self.n_envs, -1),
            seat=self._seat).clone()

    def reset(self):
        terminal = torch.ones(self.n_envs, dtype=bool, device=self.device)
        self._terminate(terminal)
        return arrdict.arrdict(terminal=torch.zeros_like(terminal), **self._observe())

    def step(self, actions):
        """Args:
            actions: (n_env, 2)-int tensor between (0, 0) and (boardsize, boardsize). Cells are indexed in row-major
            order from the top-left.
            
        Returns:

        """
        terminal = self._update_states(actions)
        old = arrdict.arrdict(reward=terminal.float(), **self._observe())
        self._seat = 1 - self._seat
        self._terminate(terminal)
        new = arrdict.arrdict(terminal=terminal, **self._observe())
        return old, new

    def state(self):
        return self._board.clone()

    @classmethod
    def plot_state(cls, state, e=0):
        board = state[e]
        width = board.shape[1]

        fig, ax = plt.subplots()
        ax.set_aspect(1)

        sin60 = np.sin(np.pi/3)
        ax.set_xlim(-1, 1.5*width - .5)
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
        ax.add_patch(mpl.patches.Polygon(np.array([tl, tr, bl, br]), linewidth=1, facecolor=black, zorder=1))
        ax.add_patch(mpl.patches.Polygon(np.array([tl, bl, tr, br]), linewidth=1, facecolor=white, zorder=1))

        radius = .5/sin60
        data_to_pixels = ax.transData.get_matrix()[0, 0]
        pixels_to_points = 1/fig.get_dpi()*72.
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

        return fig

    def display(self, e=0):
        return self.plot_state(arrdict.numpyify(self.state()), e=e)

def basic_test():
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

def open_spiel_test():
    import pyspiel

    e = 1
    ours = Hex(e+1, 11, device='cpu')
    new = ours.reset()

    theirs = pyspiel.load_game("hex")
    state = theirs.new_initial_state()
    while True:
        our_action = []
        for ee in range(ours.n_envs):
            options = (~new.obs.any(-1)[ee]).nonzero()
            our_action.append(options[torch.randint(options.size(0), ())])
        old, new = ours.step(torch.stack(our_action))

        their_action = (our_action[e] * torch.tensor([ours.boardsize, 1])).sum(-1)
        state.apply_action(their_action)
            
        if new.reset[e]:
            break
            
        our_state = ours.display(e=e, hidden=True)
        their_state = open_spiel_board(state)
        assert our_state == their_state
