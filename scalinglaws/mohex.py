import os, string, sys, subprocess
from select import select
from logging import getLogger
import shlex

log = getLogger(__name__)

class MoHex:

    def __init__(self, board_size=5, timelimit=.1, ):
        command = 'mohex --use-logfile=0'
        self._p = subprocess.Popen(shlex.split(command),
                             stdin=subprocess.PIPE, 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             text=True)
        self._logs = []
        self._log(f"# {command}\n")
        self.send(f'boardsize {board_size}')
        self.send(f'param_mohex max_time {timelimit}')

    def play(self, color, pos):
        row, col = pos
        col = chr(ord('a') + col)
        self.send(f'play {color} {col}{row}')

    def solve(self, color):
        col, row = self.send(f'genmove {color}').strip()
        col = ord(col) - ord('a')
        return int(row), col

    def display(self):
        s = self.send('showboard')
        print('\n'.join(s.splitlines()[3:-1]))

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

