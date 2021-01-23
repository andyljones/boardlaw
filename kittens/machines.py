from logging import getLogger

log = getLogger(__name__)

TYPES = {}

def register(cls):
    TYPES[cls.__name__] = cls
    return cls

def machines():
    ms = {}
    for t, cls in TYPES.items():
        for m in cls.machines():
            m['type'] = t
            ms[m['name']] = m
    return ms

def launch(j, m):
    return TYPES[m['type']].launch(j, m)

def cleanup(j):
    ms = machines()
    if j['machine'] not in ms:
        log.info(f'No cleanup for job {j["name"]} as machine "{j["machine"]}" no longer exists')
        return 

    m = ms[j['machine']]
    TYPES[m['type']].cleanup(j, m)
