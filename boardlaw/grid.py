import time
import jittens
import vast

def launch():
    for width in [1, 2, 4, 8]:
        for depth in [1, 2, 4, 8]:
            params = dict(width=width, depth=depth, boardsize=3, timelimit=15*60)
            jittens.jobs.submit(
                cmd='python -c "from boardlaw.main import *; run_jittens()" >logs.txt 2>&1',
                dir='.',
                resources={'gpu': 1},
                params=params)


def run():
    vast.jittenate(local=True)
    launch()
    while not jittens.finished():
        jittens.manage()
        time.sleep(1)