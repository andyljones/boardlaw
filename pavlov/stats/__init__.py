from .monitoring import from_run, from_run_sync
from .plotting import review, view
from .registry import KINDS as _KINDS, to_run, exists
from .gpu import gpu
from .deferral import defer, wrap
from .analysis import array, pandas, compare, plot, purge, periodic

for name, func in _KINDS.items():
    globals()[name] = wrap(func)
