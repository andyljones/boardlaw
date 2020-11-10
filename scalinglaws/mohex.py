import torch
import os, string, sys, subprocess
from select import select
from logging import getLogger
import shlex

log = getLogger(__name__)

class MoHex:

    def __init__(self, boardsize=5, timelimit=1):
        command = 'mohex --use-logfile=0'
        self._p = subprocess.Popen(shlex.split(command),
                             stdin=subprocess.PIPE, 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             text=True)
        self._logs = []
        self._log(f"# {command}\n")
        self.send(f'boardsize {boardsize}')
        # self.send(f'param_mohex use_time_management 1')
        # self.send(f'param_game game_time {timelimit/2}')
        self.send('param_mohex max_games 1')

    def send(self, cmd):
        try:
            self._log(">{cmd}\n")
            self._p.stdin.write(f'{cmd}\n')
            self._p.stdin.flush()
            return self._answer()
        except IOError:
            raise

    def _answer(self):
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
        self.send(f'play {color} {col}{row+1}')

    def solve(self, color):
        resp = self.send(f'genmove {color}').strip()
        col, row = resp[:1], resp[1:]
        col = ord(col) - ord('a')
        return int(row)-1, col

    def clear(self):
        self.send('clear_board')

    def display(self):
        s = self.send('showboard')
        print('\n'.join(s.splitlines()[3:-1]))


class MoHexAgent:

    def __init__(self, env):
        self._proxies = [MoHex(env.boardsize) for _ in range(env.n_envs)]
        self._prev_obs = torch.zeros((env.n_envs, env.boardsize, env.boardsize, 2), device=env.device)

    def __call__(self, inputs):
        seated_obs = inputs.obs
        oppo_obs = inputs.obs.transpose(1, 2).flip(3)
        obs = seated_obs.where(inputs.seats[:, None, None, None] == 0, oppo_obs)

        reset = ((obs == 0) & (self._prev_obs != 0)).any(-1).any(-1).any(-1)
        for env in reset.nonzero():
            self._proxies[env].clear()
        self._prev_obs[reset] = 0.

        new_moves = (obs != self._prev_obs).nonzero()
        for (env, row, col, seat) in new_moves:
            color = 'bw'[seat]
            self._proxies[env].play(color, (row, col))

        self._prev_obs = obs

        actions = []
        for env, seat in enumerate(inputs.seats):
            color = 'bw'[seat]
            row, col = self._proxies[env].solve(color)
            actions.append((row, col) if seat == 0 else (col, row))
            obs[env, row, col, seat] = 1.
        
        return torch.tensor(actions, dtype=torch.long, device=inputs.seats.device)

    def display(self, e=0):
        return self._proxies[e].display()

def test():
    env = hex.Hex(boardsize=3)
    black = MoHexAgent(env)
    white = MoHexAgent(env)

    inputs = env.reset()
    for _ in range(5):
        actions = black(inputs)
        responses, inputs = env.step(actions)
        actions = white(inputs)
        responses, inputs = env.step(actions)
