from . import state, machines
from logging import getLogger
from pathlib import Path

log = getLogger(__name__)

def decrement(j, m):
    for k in set(j['resources']) & set(m['resources']):
        m['resources'][k] -= j['resources'][k]

def available():
    ms = machines.machines()
    for j in state.jobs('active').values():
        if j['machine'] in ms:
            decrement(j, ms[j['machine']])
    return ms

def viable(asked, offered):
    for k in asked:
        if k not in offered:
            return False
        if asked[k] > offered[k]:
            return False
    return True

def select(j, ms):
    for m in ms.values():
        if viable(j['resources'], m['resources']):
            return m

def launch(j, m):
    log.info(f'Launching job "{j["name"]}" on machine "{m["name"]}"')
    pid = machines.TYPES[m['type']].launch(j, m)
    log.info(f'Launched with PID #{pid}')
    with state.update() as s:
        job = s['jobs'][j['name']]
        job['status'] = 'active'
        job['machine'] = m['name']
        job['process'] = pid

def dead(j):
    ms = machines.machines()
    if j['machine'] not in ms:
        log.info(f'Job "{j["name"]}" has died as the machine "{j["machine"]}" no longer exists')
        return True
    if j['process'] not in ms[j['machine']]['processes']:
        log.info(f'Job "{j["name"]}" has died as its PID #{j["process"]} is not visible on "{j["machine"]}"')
        return True
    return False

def manage():
    # Get the jobs
    for j in state.jobs('fresh').values():
        ms = available()
        m = select(j, ms)
        if m:
            launch(j, m)

    for j in state.jobs('active').values():
        if dead(j):
            with state.update() as s:
                job = s['jobs'][j['name']]
                job['status'] = 'dead'

def cleanup():
    for j in state.jobs('dead').values():
        machines.cleanup(j)
        Path(j['archive']).unlink()
        with state.update() as s:
            del s['jobs'][j['name']]

@state.mock_dir
def demo():
    from kittens import submit
    cmd = 'echo $KITTENS_GPU >"logs.txt" 2>&1'
    submit.submit(cmd, dir='.', resources={'gpu': 1})
    manage()

    assert list(state.ROOT.glob('working-dirs/*/logs.txt'))
    assert list(state.ROOT.glob('working-dirs/*/readme.md'))

    cleanup()

    assert not list(state.ROOT.glob('working-dirs/*/logs.txt'))
    assert not list(state.ROOT.glob('*.tar.gz'))
