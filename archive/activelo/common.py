import pandas as pd
import numpy as np
from rebar import arrdict

def pandify(x, names):
    if isinstance(x, np.ndarray) and x.ndim == 1:
        return pd.Series(x, names)
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return pd.DataFrame(x, names, names)
    return x

@arrdict.mapping
def numpyify(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    return x
