from . import logging, stats, runs

def monitor(run):
    import time
    with logging.from_run(run), stats.from_run(run):
        while True:
            time.sleep(.1)