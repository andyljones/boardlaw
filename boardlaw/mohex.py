import torch
import subprocess
import os
from select import select
from logging import getLogger
import shlex
import time
from . import hex
from rebar import arrdict
from tempfile import NamedTemporaryFile

log = getLogger(__name__)


def configfile(max_games=None, max_memory=None, presearch=None, 
        max_time=None, max_nodes=None, extras=[]):
    contents = []
    if max_games is not None:
        contents.append(f'param_mohex max_games {max_games}')
        if max_games < 11:
            # Gotta reduce the expand threshold when max_games is very low, else 
            # the search will never be used to update the table, and a random move'll be returned.
            contents.append(f'param_mohex expand_threshold {max_games-1}')
    if presearch is not None:
        contents.append(f'param_mohex perform_pre_search {int(presearch)}')
    if max_memory is not None:
        contents.append(f'param_mohex max_memory {int(max_memory*1e6)}')
    if max_nodes is not None:
        contents.append(f'param_mohex max_nodes {int(max_nodes)}')
    if max_time is not None:
        contents.append(f'param_mohex use_time_management 1')
        contents.append(f'param_game game_time {max_time/2}')
    contents.extend(extras)

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

def param_list():
    import re 

    mhx = MoHex()
    parts = '''param_book
    param_book_builder
    param_dfpn
    param_dfpn_db
    param_dfs
    param_dfs_db
    param_game
    param_mohex
    param_mohex_policy
    param_player_board
    param_player_ice
    param_player_vc
    param_solver_board
    param_solver_ice
    param_solver_vc'''
    for p in parts.splitlines():
        print(p)
        for line in mhx.query(p).strip().splitlines():
            try:
                m = re.match(r'([\[\]\w]+) ([\w_]+) (.*)', line)
                print(f'  {m.group(2):31s}{m.group(1):10s}{m.group(3)}')
            except:
                print(f'  {line}')


class MoHexAgent:

    def __init__(self, random=0., **kwargs):
        self._proxies = []
        self._kwargs = kwargs
        self.random = random

    def _load(self, worlds):
        if len(self._proxies) < worlds.n_envs:
            self._proxies = self._proxies + [MoHex(**self._kwargs) for _ in range(worlds.n_envs - len(self._proxies))]

        obs, seats  = worlds.obs, worlds.seats
        for e in range(len(seats)):
            self._proxies[e].load(obs[e], seats[e])

    def __call__(self, worlds):
        self._load(worlds)

        actions = torch.distributions.Categorical(probs=worlds.valid.float()).sample()
        use_mohex = torch.rand(worlds.n_envs) >= self.random

        futures = {}
        for i, (proxy, seat) in enumerate(zip(self._proxies, worlds.seats)):
            if use_mohex[i]:
                color = 'bw'[seat]
                futures[i] = proxy.solve_async(color)
        
        for i,future in futures.items():
            seat = worlds.seats[i]
            if seat == 0:
                row, col = future()
            else:
                col, row = future()
            actions[i] = worlds.boardsize*row + col
        
        return arrdict.arrdict(
            actions=actions)

    def display(self, e=0):
        return self._proxies[e].display()
    
    def to(self, device):
        assert device == 'cpu'
        return self

def check_setting(setting, common={'presearch': False, 'max_games': 1}, n_envs=9, device='cuda:1'):
    """Play a MoHex v MoHex game, with one agent with certain 'bad' parameters playing black.
    
    If the black player gets less than a perfect win-rate, hooray! You've found the 'bad' setting.
    """
    if isinstance(setting, str):
        setting = {'extras': [setting]}
    elif isinstance(setting, list):
        setting = {'extras': setting}

    from . import analysis

    world = hex.Hex.initial(n_envs=n_envs, boardsize=5, device=device)

    bad = MoHexAgent(**{**common, **setting})
    good = MoHexAgent(**common)

    trace = analysis.rollout(world, [bad, good], n_reps=1)

    mask = trace.transitions.terminal.cumsum(0).le(1)
    rates = trace.transitions.rewards[mask].eq(1).sum(0)/n_envs

    print(f'"{setting}": winrate of {rates[0]}')
    return rates[0] < 1

def check_settings():
    """This crashes vscode/ssh/whatever after 4-5 iterations"""

    params = """param_mohex
    backup_ice_info                [bool]    1
    extend_unstable_search         [bool]    1
    perform_pre_search             [bool]    1
    prior_pruning                  [bool]    1
    use_rave                       [bool]    1
    use_root_data                  [bool]    1
    virtual_loss                   [bool]    1
param_mohex_policy
    pattern_heuristic              [bool]    1
param_player_board
    backup_ice_info                [bool]    1
    use_decompositions             [bool]    1
    use_ice                        [bool]    1
    use_vcs                        [bool]    1
param_player_ice
    find_all_pattern_superiors     [bool]    1
    use_capture                    [bool]    1
    find_reversible                [bool]    1
param_player_vc
    use_patterns                   [bool]    1
    use_non_edge_patterns          [bool]    1
    incremental_builds             [bool]    1
    limit_fulls                    [bool]    1
    limit_or                       [bool]    1
param_solver_board
    backup_ice_info                [bool]    1
    use_decompositions             [bool]    1  
    use_ice                        [bool]    1  
    use_vcs                        [bool]    1  
param_solver_ice
    find_all_pattern_superiors     [bool]    1
    use_capture                    [bool]    1
    find_reversible                [bool]    1
param_solver_vc
    use_patterns                   [bool]    1
    use_non_edge_patterns          [bool]    1
    incremental_builds             [bool]    1
    limit_fulls                    [bool]    1
    limit_or                       [bool]    1"""

    results = {}
    category = None
    for l in params.splitlines():
        if not l.startswith('  '):
            category = l.strip()
        elif l.strip().startswith('#'):
            print(f'Skipping "{l.strip()}"')
            continue
        else:
            print(f'Trying "{l.strip()}"')
            name = l.strip().split(' ')[0]
            results[f'{category} {name}'] = check_setting(f'{category} {name} 0')
            time.sleep(5)

    return results

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
