import torch
import numpy as np
from rebar import arrdict
from .. import heads
import boardlaw

def observe(board, seats):
    obs = torch.stack([board == 1, board == 2], -2).float()
    obs[seats == 1] = obs[seats == 1].flip(0).flip(2)
    return obs

def step(board, seats, actions):
    pass

class Breakthrough(arrdict.namedarrtuple(fields=('board', 'seats'))):

    @classmethod
    def initial(cls, n_envs, boardsize=8, device='cuda'):
        boardsize = boardsize if isinstance(boardsize, tuple) else (boardsize, boardsize)
        assert boardsize[0] >= 4

        board = torch.full((n_envs, *boardsize), 0, device=device, dtype=torch.uint8)
        board[:, :2] = 1
        board[:, -2:] = 2

        return cls(
            board=board,
            seats=torch.full((n_envs,), 0, device=device, dtype=torch.int))

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
        self.action_space = heads.Masked(3*self.boardsize*self.boardsize)

        self._obs = None
        self._valid = None 


    @property
    def obs(self):
        if self._obs is None:
            self._obs = observe(self.board, self.seats)
        return self._obs

    @property
    def valid(self):
        if self._valid is None:
            shape = self.board.shape[:-2]
            self._valid = (self.obs == 0).all(-1).reshape(*shape, -1)
        return self._valid

    def step(self, actions):
        """Args:
            actions: (n_env, 2, 3)-int tensor between (0, 0) and (boardsize, boardsize, 3). Cells are indexed in row-major
                    order from the top-left, and the action goes ahead-left/ahead/ahead-right
            
        Returns:

        """
        if self.board.ndim != 3:
            #TODO: Support stepping arbitrary batchings. Only needs a reshaping.
            raise ValueError('You can only step a board with a single batch dimension')

        assert (0 <= actions).all(), 'You passed a negative action'
        if actions.ndim == 3:
            actions = actions[..., 0]*3*self.boardsize + actions[:, 1]*3 + actions[2]

        assert actions.shape == (self.n_envs,)
        assert self.valid.gather(1, actions[:, None]).squeeze(-1).all()

        new_board = self.board.clone()
        rewards = cuda.step(new_board, self.seats.int(), actions.int())
        terminal = (rewards > 0).any(-1)

        new_board[terminal] = 0

        new_seat = 1 - self.seats
        new_seat[terminal] = 0

        new_world = type(self)(board=new_board, seats=new_seat)

        transition = arrdict.arrdict(
            terminal=terminal, 
            rewards=rewards)
        return new_world, transition

    def __getitem__(self, x):
        # Just exists for profiling
        return super().__getitem__(x)

    def __setitem__(self, x, y):
        # Just exists for profiling
        return super().__setitem__(x, y)

def open_spiel_board(state):
    # state ordering taken from hex.h 
    strs = 'bw.'
    board = np.array(state.observation_tensor()).reshape(3, 8, 8).argmax(0)
    strs = np.vectorize(strs.__getitem__)(board)
    return '\n'.join(' '.join(r) for i, r in enumerate(strs))

def open_spiel_display_str(env, e):
    board = env.board[e].clone()
    strings = np.vectorize('.bw'.__getitem__)(board.cpu().numpy())
    return '\n'.join(' '.join(r) for i, r in enumerate(strings))

def test_open_spiel():
    """https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/breakthrough.cc"""
    import pyspiel

    e = 10
    ours = Breakthrough.initial(64, 8, 'cpu')

    theirs = pyspiel.load_game("breakthrough")
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