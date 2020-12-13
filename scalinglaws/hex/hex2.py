from rebar import arrdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from .. import heads
from . import cuda

CHARS = '.bwTBLR'
ORDS = {c: i for i, c in enumerate(CHARS)}

class Hex(arrdict.namedarrtuple(fields=('board', 'seat'))):

    @classmethod
    def initial(cls, n_envs, boardsize=11, device='cuda'):
        # As per OpenSpiel and convention, black plays first.
        return cls(
            board=torch.full((n_envs, boardsize, boardsize), 0, device=device, dtype=torch.uint8),
            seat=torch.full((n_envs,), 0, device=device, dtype=torch.int))

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

    @property
    def obs(self):
        if self._obs is None:
            black_view = torch.stack([
                torch.stack([self.board == ORDS[s] for s in 'bTB']).any(0),
                torch.stack([self.board == ORDS[s] for s in 'wLR']).any(0)], -1).float()

            # White player sees a transposed board
            white_view = black_view.transpose(-3, -2).flip(-1)
            self._obs = black_view.where(self.seat[..., None, None, None] == 0, white_view)
        return self._obs

    @property
    def valid(self):
        shape = self.board.shape[:-2]
        return (self.obs == 0).all(-1).reshape(*shape, -1)

    @property
    def seats(self):
        return self.seat

    def step(self, actions):
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

        new_board = self.board.clone()
        rewards = cuda.step(new_board, self.seats.int(), actions.int())
        terminal = (rewards > 0).any(-1)

        new_board[terminal] = 0

        new_seat = 1 - self.seat
        new_seat[terminal] = 0

        new_world = type(self)(board=new_board, seat=new_seat)

        transition = arrdict.arrdict(
            terminal=terminal, 
            rewards=rewards)
        return new_world, transition

    @classmethod
    def plot_worlds(cls, worlds, e=None, ax=None):
        e = (0,)*(worlds.board.ndim-2) if e is None else e
        board = worlds.board[e]
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
        colors = ['tan', black, white, black, black, white, white] 
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

    def display(self, e=None):
        ax = self.plot_worlds(arrdict.numpyify(arrdict.arrdict(self)), e=e)
        plt.close(ax.figure)
        return ax


def regression_test():
    from . import Hex, hex2
    kwargs = dict(n_envs=1, boardsize=3)
    old = Hex.initial(**kwargs)
    new = hex2.Hex.initial(**kwargs)

    history = []
    for i in range(1000):
        actions = torch.distributions.Categorical(probs=old.valid.float()).sample()
        history.append(actions)
        
        oldn, oldt = old.step(actions)
        newn, newt = new.step(actions)
        
        torch.testing.assert_allclose(oldn.obs, newn.obs)
        torch.testing.assert_allclose(oldt.rewards, newt.rewards)
        torch.testing.assert_allclose(oldt.terminal, newt.terminal)
        
        if oldt.terminal.any():
            history = []
        
        old, new = oldn, newn

def test_bug():
    worlds = Hex.initial(n_envs=1, boardsize=3)
    actions = torch.tensor([5, 5, 6, 1], device=worlds.device)
    for a in actions:
        worlds, transitions = worlds.step(a[None])
    torch.testing.assert_allclose(worlds.board[0], torch.tensor([
        [0, 0, 0],
        [5, 0, 1],
        [4, 2, 0]], device=worlds.device))