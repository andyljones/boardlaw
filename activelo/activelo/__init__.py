from .solvers import Solver, solve
from .suggestions import suggest

import numpy as np
from logging import getLogger

log = getLogger(__name__)

def safe_suggest(n, w, G):
    try:
        soln = solve(n, w)
        log.info(f'Fitted a posterior, {(soln.σd**2).mean()**.5:.2f}σd over {n.shape[0]} agents')
        suggestion = suggest(soln, G)
        log.info(f'Suggestion is {suggestion}')
        return suggestion
    except ValueError:
        log.warn('Solver failed; making a random suggestion')
        return tuple(np.random.randint(0, n.shape[0], (2,)))