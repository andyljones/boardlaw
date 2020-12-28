from . import logs, stats, runs, tests

def monitor(run=-1):
    import time
    with logs.from_run(run), stats.from_run(run):
        while True:
            time.sleep(.1)

@tests.mock_dir
def demo_monitor():
    import time
    from logging import getLogger

    log = getLogger(__name__)

    run = runs.new_run()
    with stats.from_run(run), logs.from_run(run):
        with stats.to_run(run), logs.to_run(run):
            
            stats.mean('test', 1, 2)
            log.info('hello')

            while True:
                time.sleep(.1)