import torch
import numpy as np
from rebar import arrdict

class Hex:
    """Based on `OpenSpiel's implementation <https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/hex.cc>`_.
    """

    # Empty, 
    # black, black win, black-north-connected, black-south-connected
    # white, white win, white-west-connected, white-east-connected
    _STRINGS = '.bB^vwW<>'
    

    def __init__(self, n_envs=1, boardsize=11, device='cuda'):
        self.n_envs = n_envs
        self._boardsize = boardsize
        self._device = torch.device(device)

        self._STATES = {s: torch.tensor(i, dtype=torch.int, device=device) for i, s in enumerate(self._STRINGS)}

        self._IS_EDGE = {
            '^': lambda idxs: idxs[..., 0] == 0,
            'v': lambda idxs: idxs[..., 0] == boardsize-1,
            '<': lambda idxs: idxs[..., 1] == 0,
            '>': lambda idxs: idxs[..., 1] == boardsize-1}

        self._NEIGHBOURS = torch.tensor([(-1, 0), (-1, +1), (0, -1), (0, +1), (+1, -1), (+1, +0)], device=device, dtype=torch.long)

        self._board = torch.full((n_envs, boardsize, boardsize), 0, device=device, dtype=torch.int)

        # As per OpenSpiel and convention, black plays first.
        self._player = torch.full((n_envs,), 0, device=device, dtype=torch.int)
        self._envs = torch.arange(self.n_envs, device='cuda')

    def _states(self, idxs, val=None):
        rows, cols = idxs[..., 0], idxs[..., 1]
        envs = self._envs[(slice(None),) + (None,)*(idxs.ndim-2)].expand_as(rows)
        if val is None:
            return self._board[envs, rows, cols]
        else:
            self._board[envs, rows, cols] = val

    def _neighbours(self, idxs):
        return self._states((idxs[:, None, :] + self._NEIGHBOURS).clamp(0, self._boardsize-1))

    def _update_states(self, actions):
        assert (self._states(actions) == 0).all(), 'One of the actions is to place a token on an already-occupied cell'

        neighbours = self._neighbours(actions)

        black = self._player == 0
        white = self._player == 1
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

        reset = ((new_state == self._STATES['B']) | (new_state == self._STATES['W']))
        self._board[reset] = self._STATES['.']

        return reset

    def step(self, actions):
        """Args:
            actions: (n_env, 2)-int tensor between (0, 0) and (boardsize, boardsize). Cells are indexed in row-major
            order from the top-left.
            
        Returns:

        """

        reset = self._update_states(actions)
        reward = reset.float()

        obs = torch.stack([
            torch.stack([self._board == self._STATES[s] for s in 'b^vB']).any(0),
            torch.stack([self._board == self._STATES[s] for s in 'w<>W']).any(0)], -1)

        old = arrdict.arrdict(
            player=self._player,
            reward=reward).clone()

        self._player = 1 - self._player

        new = arrdict.arrdict(
                obs=obs,
                reset=reset,
                player=self._player).clone()

        return old, new

    def display(self, e=0):
        strings = np.vectorize(self._STRINGS.__getitem__)(self._board[e].cpu().numpy())
        print('\n'.join(''.join(r) for r in strings))
