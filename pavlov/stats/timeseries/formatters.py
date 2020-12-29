import numpy as np
import pandas as pd
from ... import tests
from .. import registry

def final_row(reader, rule):
    # Offset slightly into the past, else by the time the resample actually happens you're 
    # left with an almost-empty last interval.
    offset = f'{(tests.time() % 60) - 5}s'
    df = reader.pandas()
    df.index = df.index + reader._created
    resampled = reader.resampler(**df, rule=rule, offset=offset)
    resampled = resampled.iloc[:-1].ffill(limit=1) # Drop that almost-empty last interval
    if len(resampled) > 0: 
        return resampled.iloc[-1]

def channel(reader):
    return registry.parse_prefix(reader.prefix).channel

def simple(reader, rule):
    final = final_row(reader, rule)
    if final is None:
        return []
    final = final.item()
    if isinstance(final, int):
        return [(channel(reader), f'{final:<6g}')]
    if isinstance(final, float):
        return [(channel(reader), f'{final:<6g}')]
    else:
        raise ValueError() 

def percent(reader, rule):
    final = final_row(reader, rule)
    if final is None:
        return []
    return [(channel(reader), f'{final.item():.2%}')]

def confidence(reader, rule):
    final = final_row(reader, rule)
    if final is None:
        return []
    return [(channel(reader), f'{final.μ:.2f}±{2*final.σ:.2f}')]


