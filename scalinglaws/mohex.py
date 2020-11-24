import torch
import subprocess
import os
from select import select
from logging import getLogger
import shlex
from . import hex
from rebar import arrdict
from tempfile import NamedTemporaryFile

log = getLogger(__name__)

def configfile(max_games=None, max_memory=None, presearch=None, max_time=None, max_nodes=None):
    contents = []
    if max_games is not None:
        contents.append(f'param_mohex max_games {max_games}')
    if presearch is not None:
        contents.append(f'param_mohex perform_pre_search {int(presearch)}')
    if max_memory is not None:
        contents.append(f'param_mohex max_memory {int(max_memory*1e6)}')
    if max_nodes is not None:
        contents.append(f'param_mohex max_nodes {int(max_nodes)}')
    if max_time is not None:
        contents.append(f'param_mohex use_time_management 1')
        contents.append(f'param_game game_time {max_time/2}')

    with NamedTemporaryFile('w', delete=False, prefix='mohex-config-') as f:
        f.write('\n'.join(contents))

    return f.name

def to_notation(subscript):
    row, col = subscript
    col = chr(ord('a') + col)
    return f'{col}{row+1}'

def as_sgf(obs, seat):
    """Example: https://github.com/cgao3/benzene-vanilla-cmake/blob/master/regression/sgf/opening/5x5a3.sgf
    Wiki: https://en.wikipedia.org/wiki/Smart_Game_Format
    """
    assert obs.ndim == 3, 'Observations must be a (S, S, 2) stack of piece indicators'
    size = obs.size(0)
    obs = obs.transpose(0, 1).flip(2) if seat == 1 else obs

    positions = {'B': obs[..., 0].nonzero(as_tuple=False), 'W': obs[..., 1].nonzero(as_tuple=False)}
    moves = []
    for colour, posns in positions.items():
        for pos in posns:
            moves.append(f'{colour}[{to_notation(pos)}]')
    return f"(;AP[HexGui:0.2]FF[4]GM[11]SZ[{size}];{';'.join(moves)})"

class MoHex:

    def __init__(self, *args, **kwargs):
        filename = configfile(*args, **kwargs) 
        command = f'mohex --use-logfile=0 --config={filename}'
        self._p = subprocess.Popen(shlex.split(command),
                            stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            text=True)

        log.debug(f"# {command}")

    def _log_std_err(self):
        list = select([self._p.stderr], [], [], 0)[0]
        for s in list:
            s = os.read(s.fileno(), 8192).decode()
            for l in s.splitlines():
                log.debug(l)

    def answer(self):
        self._log_std_err()
        lines = []
        while True:
            line = self._p.stdout.readline()
            if line == "":
                # Program died
                self._log_std_err()
                raise IOError('MoHex returned an empty line')
            log.debug(f'<{line.strip()}')
            done = (line == "\n")
            if done:
                break
            else:
                lines += line

        answer = ''.join(lines)
        if answer[0] != '=':
            raise ValueError(answer[2:].strip())
        if len(lines) == 1:
            return answer[1:].strip()
        return answer[2:]

    def send(self, cmd):
        log.debug(f">{cmd}")
        self._p.stdin.write(f'{cmd}\n')
        self._p.stdin.flush()

        return self.answer

    def query(self, cmd):
        return self.send(cmd)()

    def load(self, obs, seat):
        sgf = as_sgf(obs, seat)
        with NamedTemporaryFile('w') as f:
            f.write(sgf)
            f.flush()
            self.query(f'loadsgf {f.name}')

    def boardsize(self, size):
        self.query(f'boardsize {size}')

    def play(self, color, pos):
        self.query(f'play {color} {to_notation(pos)}')

    def solve_async(self, color):
        f = self.send(f'reg_genmove {color}')

        def future():
            resp = f().strip()
            col, row = resp[:1], resp[1:]
            col = ord(col) - ord('a')
            return int(row)-1, col
        
        return future

    def solve(self, color):
        return self.solve_async(color)()

    def clear(self):
        self.query('clear_board')

    def display(self):
        s = self.query('showboard')
        print('\n'.join(s.splitlines()[3:-1]))

class MoHexAgent:

    def __init__(self, **kwargs):
        self._proxies = []
        self._kwargs = kwargs

    def _load(self, worlds):
        if len(self._proxies) < worlds.n_envs:
            self._proxies = self._proxies + [MoHex(**self._kwargs) for _ in range(worlds.n_envs - len(self._proxies))]

        obs, seats  = worlds.obs, worlds.seats
        for e in range(len(seats)):
            self._proxies[e].load(obs[e], seats[e])

    def __call__(self, worlds):
        self._load(worlds)

        futures = []
        for proxy, seat in zip(self._proxies, worlds.seats):
            color = 'bw'[seat]
            futures.append(proxy.solve_async(color))
        
        actions = []
        for future, seat in zip(futures, worlds.seats):
            row, col = future()
            actions.append((row, col) if seat == 0 else (col, row))

        actions = torch.tensor(actions, dtype=torch.long, device=worlds.device)

        # To linear indices
        actions = actions[:, 0]*worlds.boardsize + actions[:, 1]
        
        return arrdict.arrdict(
            actions=actions)

    def display(self, e=0):
        return self._proxies[e].display()

def test():
    worlds = hex.Hex.initial(1, boardsize=5)
    agents = MoHexAgent(max_games=1000)

    for _ in range(10):
        decisions = agents(worlds)
        worlds, transitions = worlds.step(decisions.actions)

def benchmark(n=16, T=10, **kwargs):
    # kwargs/rate
    # 16, Defaults: 1.4
    # 16, presearch=False, max_games=1: 13.2
    # 16, presearch=False, max_games=1, max_memory=1: 12
    # 16, presearch=False, max_time=1: 6
    # 16, max_time=1: 6
    # 16, max_games=1, 7.6
    # 16, max_games=10, 7.2
    # 16, max_games=100, 6.6
    # 16, max_games=1000, 6.0
    # 16, max_games=10000, 3.9

    import aljpy 
    import pandas as pd

    worlds = hex.Hex.initial(n_envs=n, boardsize=11)
    agents = MoHexAgent(**kwargs)

    # Prime it
    agents(worlds)

    with aljpy.timer() as timer:
        moves = 0
        for _ in range(T):
            decisions = agents(worlds)
            worlds, transitions = worlds.step(decisions.actions)
            moves += worlds.n_envs
    s = pd.Series({'n_envs': n, 'runtime': timer.time(), 'samples': moves})
    s['rate'] = s.samples/s.runtime
    return s