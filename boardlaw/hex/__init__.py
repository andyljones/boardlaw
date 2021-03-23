from rebar import arrdict, profiling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from .. import heads
from . import cuda

CHARS = '.bwTBLR'
ORDS = {c: i for i, c in enumerate(CHARS)}

def color_board(board, colors='obs'):
    black = (0, 0, .4)
    white = (0, 0, .8)
    tan = (.07, .4, .8)
    if colors == 'obs':
        colors = [tan, black, white, black, black, white, white] 
    elif colors == 'board':
        colors = [tan, black, white, (.16, .2, .4), (.33, .2, .4), (.66, .2, .8), (.72, .2, .8)]
    colors = np.stack([mpl.colors.hsv_to_rgb(c) for c in colors])
    colors = colors[board]
    return colors

def color_obs(obs):
    keyed = np.zeros_like(obs[..., 0], dtype=int)
    keyed[obs[..., 0] == 1.] = 1
    keyed[obs[..., 1] == 1.] = 2
    return color_board(keyed)

def _hex_centers(width):
    # Find Hex centers
    sin60 = np.sin(np.pi/3)
    rows, cols = np.indices((width, width))
    return np.stack([
        cols + .5*np.arange(width)[:, None],
        # Hex centers are 1 apart, so distances between rows are sin(60)
        sin60*(width - 1 - rows)], -1).reshape(-1, 2)

def _hex_size(ax):
    # Generate hexes
    sin60 = np.sin(np.pi/3)
    radius = .5/sin60
    data_to_pixels = ax.transData.get_matrix()[0, 0]
    pixels_to_points = 1/ax.figure.get_dpi()*72.
    return np.pi*(data_to_pixels*pixels_to_points*radius)**2

def _edge(width, k=.2):
    sin30 = np.sin(np.pi/6)
    cos30 = np.cos(np.pi/6)
    sin60 = np.sin(np.pi/3)
    cos60 = np.cos(np.pi/3)

    left_center = np.array([0, 0])
    left_point = (1/2/cos30 + k/cos30)*np.array([-cos30, sin30])
    left = np.stack([left_center, left_point])

    top = np.full((2*width-1, 2), np.nan)
    top[::2, 0] = np.arange(width)
    top[::2, 1] = np.full(width, 1/2/cos30 + k/cos30)

    top[1::2, 0] = .5 + np.arange(width-1)
    top[1::2, 1] = np.full(width-1, sin30/2/cos30 + k/cos30)

    right_point = np.array([width - 1, 0]) + (1/2 + k)*np.array([cos60, sin60])
    right_center = np.array([width - 1, 0])
    right = np.stack([right_point, right_center])

    return np.concatenate([left, top, right])

def _edges(coords, black, white, width):
    # Generate board edges
    rots = [0, 120, 180, 300]
    origins = [coords[0], coords[0], coords[-1], coords[-1]]
    cols = [black, white, black, white]
    patches = []
    for rot, origin, color in zip(rots, origins, cols):
        rot = np.pi/180*rot
        rot = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        ref = 1 if color == black else -1
        rot[:, 1] = ref*rot[:, 1] 
        
        points = _edge(width)*ref @ rot + origin
        patches.append(mpl.patches.Polygon(points, linewidth=1, edgecolor='k', facecolor=color, zorder=1))
    return patches


def plot_board(colors, ax=None, black='dimgray', white='lightgray', rotate=False):
    ax = plt.subplots()[1] if ax is None else ax
    ax.set_aspect(1)

    width = colors.shape[0]

    sin60 = np.sin(np.pi/3)
    ax.set_xlim(-1.5, 1.5*width)
    ax.set_ylim(-sin60, sin60*width)

    size = _hex_size(ax)
    coords = _hex_centers(width)

    hexes = mpl.collections.RegularPolyCollection(
                    numsides=6, 
                    rotation=np.pi/180*30 if rotate else 0,
                    sizes=(size,)*len(coords),
                    offsets=coords, 
                    facecolors=colors.reshape(-1, colors.shape[-1]), 
                    edgecolor='k', 
                    linewidths=1, 
                    transOffset=ax.transData,
                    zorder=2)
    ax.add_collection(hexes)

    for patch in _edges(coords, black, white, width):
        ax.add_patch(patch)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

class Hex(arrdict.namedarrtuple(fields=('board', 'seats'))):

    @classmethod
    def initial(cls, n_envs, boardsize=11, device='cuda'):
        # As per OpenSpiel and convention, black plays first.
        return cls(
            board=torch.full((n_envs, boardsize, boardsize), 0, device=device, dtype=torch.uint8),
            seats=torch.full((n_envs,), 0, device=device, dtype=torch.int))

    @profiling.nvtx
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.board, torch.Tensor):
            # Need this conditional to deal with the case where we're calling a method like `self.clone()`, and the
            # intermediate arrdict generated is full of methods, which will break this here init function.
            return 

        self.n_seats = 2
        self.n_envs = self.board.shape[0]
        self.boardsize = self.board.shape[1]
        self.device = self.board.device

        self.obs_space = heads.Tensor((self.boardsize, self.boardsize, 2))
        self.action_space = heads.Masked(self.boardsize*self.boardsize)

        self._obs = None
        self._valid = None 

    @property
    def obs(self):
        if self._obs is None:
            self._obs = cuda.observe(self.board, self.seats)
        return self._obs

    @property
    def valid(self):
        if self._valid is None:
            shape = self.board.shape[:-2]
            self._valid = (self.obs == 0).all(-1).reshape(*shape, -1)
        return self._valid

    @profiling.nvtx
    def step(self, actions, reset=True):
        """Args:
            actions: (n_env, 2)-int tensor between (0, 0) and (boardsize, boardsize). Cells are indexed in row-major
            order from the top-left.
            
        Returns:

        """
        if self.board.ndim != 3:
            #TODO: Support stepping arbitrary batchings. Only needs a reshaping.
            raise ValueError('You can only step a board with a single batch dimension')

        assert (0 <= actions).all(), 'You passed a negative action'
        if actions.ndim == 2:
            actions = actions[..., 0]*self.boardsize + actions[:, 1]

        assert actions.shape == (self.n_envs,)
        assert self.valid.gather(1, actions[:, None]).squeeze(-1).all()

        new_board = self.board.clone()
        rewards = cuda.step(new_board, self.seats.int(), actions.int())
        terminal = (rewards > 0).any(-1) if reset else torch.full((self.n_envs,), False, device=self.device)

        new_board[terminal] = 0

        new_seat = 1 - self.seats
        new_seat[terminal] = 0

        new_world = type(self)(board=new_board, seats=new_seat)

        transition = arrdict.arrdict(
            terminal=terminal, 
            rewards=rewards)
        return new_world, transition

    @profiling.nvtx
    def __getitem__(self, x):
        # Just exists for profiling
        return super().__getitem__(x)

    @profiling.nvtx
    def __setitem__(self, x, y):
        # Just exists for profiling
        return super().__setitem__(x, y)

    @classmethod
    def plot_worlds(cls, worlds, e=None, ax=None, colors='obs', **kwargs):
        e = (0,)*(worlds.board.ndim-2) if e is None else e
        board = worlds.board[e]

        ax = plt.subplots()[1] if ax is None else ax

        colors = color_board(board, colors)
        plot_board(colors, ax, **kwargs)

        return ax.figure

    def display(self, e=None, **kwargs):
        ax = self.plot_worlds(arrdict.numpyify(arrdict.arrdict(self)), e=e, **kwargs)
        plt.close(ax.figure)
        return ax

class Solitaire(Hex):
    """One-player Hex"""

    @classmethod
    def initial(cls, *args, seat=0, **kwargs):
        worlds = super().initial(*args, **kwargs)
        if seat == 1:
            raise ValueError('Can\'t do seat #1 right now')
        return worlds

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_seats = 1

    def step(self, actions):
        worlds, transitions = super().step(actions)

        # Might be that the move just made wins the game, in which case we need to 
        # step the world until we get to the same seat again.
        while True:
            mask = (worlds.seats != self.seats)
            if not mask.any():
                break
            worlds[mask], other = self._play(worlds[mask])
            transitions.rewards[mask] += other.rewards
            transitions.terminal[mask] |= other.terminal

        envs = torch.arange(self.n_envs, device=self.device)
        transitions['rewards'] = transitions.rewards[envs, self.seats.long()][:, None]
        return worlds, transitions

class Lazy(Solitaire):
    """Opponent plays the first available action"""

    @classmethod
    def _play(cls, worlds):
        n_actions = worlds.valid.size(1)
        actions = torch.arange(n_actions, device=worlds.device)[None, :].expand_as(worlds.valid).clone()
        actions[~worlds.valid] = n_actions
        return Hex.step(worlds, actions.min(-1).values)

class Random(Solitaire):
    """Opponent plays a random action"""

    @classmethod
    def _play(cls, worlds):
        actions = torch.distributions.Categorical(probs=worlds.valid.float()).sample()
        return Hex.step(worlds, actions)


def test_bug():
    worlds = Hex.initial(n_envs=1, boardsize=3)
    actions = torch.tensor([5, 5, 6, 1], device=worlds.device)
    for a in actions:
        worlds, transitions = worlds.step(a[None])
    torch.testing.assert_allclose(worlds.board[0], torch.tensor([
        [0, 0, 0],
        [5, 0, 1],
        [4, 2, 0]], device=worlds.device))

def test_bug_2():
    worlds = Hex.initial(n_envs=1, boardsize=3)
    worlds.board[:] = torch.tensor([
        [0, 6, 6],
        [1, 1, 1],
        [0, 2, 0]], device=worlds.device, dtype=torch.uint8)
    worlds.seats[:] = 0

    worlds, transitions = worlds.step(torch.tensor([6], device=worlds.device))

    torch.testing.assert_allclose(worlds.board[0], torch.tensor([
        [0, 6, 6],
        [4, 4, 4],
        [4, 2, 0]], device=worlds.device))