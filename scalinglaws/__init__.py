import torch
import numpy as np
from rebar import arrdict

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
        self._envs = torch.arange(self.n_envs, device=device)

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
    
    def _reset(self, reset):
        self._board[reset] = self._STATES['.']
        self._player[reset] = 0

    def _neighbours(self, idxs):
        if idxs.size(1) == 3:
            neighbours = self._neighbours(idxs[:, 1:])
            envs = idxs[:, None, [0]].expand(-1, len(self._NEIGHBOURS), 1)
            return torch.cat([envs, neighbours], 2)
        return (idxs[:, None, :] + self._NEIGHBOURS).clamp(0, self._boardsize-1)

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
            idxs = unique(neighbour_idxs[possible], [self._boardsize, self._boardsize])

    def _update_states(self, actions):
        assert (self._states(actions) == 0).all(), 'One of the actions is to place a token on an already-occupied cell'

        neighbours = self._states(self._neighbours(actions))

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
        self._flood(actions)

        reset = ((new_state == self._STATES['B']) | (new_state == self._STATES['W']))

        return reset

    def _observe(self):
        obs = torch.stack([
            torch.stack([self._board == self._STATES[s] for s in 'b^vB']).any(0),
            torch.stack([self._board == self._STATES[s] for s in 'w<>W']).any(0)], -1)

        return arrdict.arrdict(
            obs=obs,
            player=self._player).clone()

    def reset(self):
        reset = torch.ones(self.n_envs, dtype=bool, device=self._device)
        self._reset(reset)
        return arrdict.arrdict(reset=reset, **self._observe())

    def step(self, actions):
        """Args:
            actions: (n_env, 2)-int tensor between (0, 0) and (boardsize, boardsize). Cells are indexed in row-major
            order from the top-left.
            
        Returns:

        """
        reset = self._update_states(actions)
        old = arrdict.arrdict(reward=reset.float(), **self._observe())
        self._player = 1 - self._player
        self._reset(reset)
        new = arrdict.arrdict(reset=reset, **self._observe())
        return old, new

    def display(self, e=0, hidden=False):
        if hidden:
            strs = self._STRINGS
            board = self._board[e].clone()
        else:
            strs = {int(self._STATES['b']): '◉', int(self._STATES['w']): '◯', int(self._STATES['.']): '·'}
            board = self._colours(self._board[e])
        strings = np.vectorize(strs.__getitem__)(board.cpu().numpy())
        return '\n'.join(' '*i + ' '.join(r) for i, r in enumerate(strings))

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

        their_action = (our_action[e] * torch.tensor([ours._boardsize, 1])).sum(-1)
        state.apply_action(their_action)
            
        if new.reset[e]:
            break
            
        our_state = ours.display(e=e, hidden=True)
        their_state = open_spiel_board(state)
        assert our_state == their_state
