from . import submit, state, machines

def decrement(j, m):
    for k in j['resources'] & m['resources']:
        m['resources'][k] -= j['resources'][k]

def available():
    ms = machines.machines()
    for sub in submit.jobs('active'):
        if sub['machine'] in ms:
            decrement(ms[sub['machine']], sub)
    return ms

def viable(asked, offered):
    for k in asked:
        if k not in offered:
            return False
        if asked[k] > offered[k]:
            return False
    return True

def select(j, ms):
    for m in ms:
        if viable(j['resources'], m['resources']):
            return m

def launch(j, m):
    pid = machines.TYPES[m['type']].launch(j)
    with state.update() as s:
        job = s['jobs'][j['name']]
        job['status'] = 'active'
        job['machine'] = m['name']
        job['process'] = pid

def dead(j):
    ms = machines.machines()
    if j['machine'] not in ms:
        return True
    if j['process'] not in ms[j['machine']]['processes']:
        return True
    return False

def manage():
    # Get the jobs
    for j in state.jobs('fresh'):
        ms = available()
        m = select(j, ms)
        launch(j, m)

    for j in state.jobs('active'):
        if dead(j):
            with state.update() as s:
                job = s['jobs'][j['name']]
                job['status'] = 'dead'
