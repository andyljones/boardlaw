from . import logs, stats, runs

def monitor(run):
    import time
    with logs.from_run(run), stats.from_run(run):
        while True:
            time.sleep(.1)