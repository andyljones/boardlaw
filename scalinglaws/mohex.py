import torch
import subprocess
import os
from select import select
from logging import getLogger
import shlex
from . import hex
from rebar import arrdict

log = getLogger(__name__)

class MoHex:

    def __init__(self, boardsize=5, max_games=1000, pre_search=True):
        command = 'mohex --use-logfile=0'
        self._p = subprocess.Popen(shlex.split(command),
                             stdin=subprocess.PIPE, 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             text=True)
        self._logs = []
        self._log(f"# {command}\n")
        self.query(f'boardsize {boardsize}')
        self.query(f'param_mohex max_games {max_games}')
        self.query(f'param_mohex perform_pre_search {int(pre_search)}')

    def answer(self):
        self._log_std_err()
        lines = []
        while True:
            line = self._p.stdout.readline()
            if line == "":
                # Program died
                self._log_std_err()
                raise IOError('MoHex returned an empty line')
            self._log(f'<{line}')
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
        self._log(">{cmd}\n")
        self._p.stdin.write(f'{cmd}\n')
        self._p.stdin.flush()

        return self.answer

    def query(self, cmd):
        return self.send(cmd)()

    def _log(self, message):
        self._logs.append(message)
        log.debug(f'MoHex: {message}')

    def _log_std_err(self):
        list = select([self._p.stderr], [], [], 0)[0]
        for s in list:
            self._log(os.read(s.fileno(), 8192))

    def play(self, color, pos):
        row, col = pos
        col = chr(ord('a') + col)
        self.query(f'play {color} {col}{row+1}')

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

    def __init__(self):
        self._proxies = None
        self._prev_obs = None

    def __call__(self, world):
        if self._proxies is None:
            self._proxies = [MoHex(world.boardsize) for _ in range(world.n_envs)]
            self._prev_obs = torch.zeros((world.n_envs, world.boardsize, world.boardsize, 2), device=world.device)

        seated_obs = world.obs
        oppo_obs = world.obs.transpose(1, 2).flip(3)
        obs = seated_obs.where(world.seats[:, None, None, None] == 0, oppo_obs)

        reset = ((obs == 0) & (self._prev_obs != 0)).any(-1).any(-1).any(-1)
        for env in reset.nonzero(as_tuple=False):
            self._proxies[env].clear()
        self._prev_obs[reset] = 0.

        new_moves = (obs != self._prev_obs).nonzero(as_tuple=False)
        for (env, row, col, seat) in new_moves:
            color = 'bw'[seat]
            self._proxies[env].play(color, (row, col))

        self._prev_obs = obs

        futures = []
        for proxy, seat in zip(self._proxies, world.seats):
            color = 'bw'[seat]
            futures.append(proxy.solve_async(color))
        
        actions = []
        for future, seat in zip(futures, world.seats):
            row, col = future()
            actions.append((row, col) if seat == 0 else (col, row))

        actions = torch.tensor(actions, dtype=torch.long, device=world.device)

        # To linear indices
        boardsize = self._prev_obs.size(1)
        actions = actions[:, 0]*boardsize + actions[:, 1]
        
        return arrdict.arrdict(
            actions=actions)

    def display(self, e=0):
        return self._proxies[e].display()

def test():
    env = hex.Hex.initial(boardsize=3)
    black = MoHexAgent(env)
    white = MoHexAgent(env)

    inputs = env.reset()
    for _ in range(5):
        actions = black(inputs)
        responses, inputs = env.step(actions)
        actions = white(inputs)
        responses, inputs = env.step(actions)

def benchmark():
    import aljpy 
    import pandas as pd

    results = []
    for n in [1, 2, 4, 8, 16]:
        env = hex.Hex.initial(n_envs=n, boardsize=11)
        black = MoHexAgent(env)
        white = MoHexAgent(env)

        with aljpy.timer() as timer:
            inputs = env.reset()
            moves = 0
            for _ in range(5):
                actions = black(inputs)
                responses, inputs = env.step(actions)
                actions = white(inputs)
                responses, inputs = env.step(actions)
                moves += 2*inputs.valid.size(0)
            results.append({'n_envs': n, 'runtime': timer.time(), 'samples': moves}) 
            print(results[-1])

    results = pd.DataFrame(results)