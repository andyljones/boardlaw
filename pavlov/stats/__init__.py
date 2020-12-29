from .monitoring import from_run, from_run_sync
from .plotting import review, view
from .registry import to_run, KINDS, array, pandas, compare, exists
from .gpu import gpu
from .deferral import defer, wrap

for name, func in KINDS.items():
    globals()[name] = wrap(func)
