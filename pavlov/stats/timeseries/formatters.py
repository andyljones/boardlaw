from ... import tests

def final_row(reader, rule):
    df = reader.pandas()

    # Offset slightly into the future, else by the time the resample actually happens you're 
    # left with an almost-empty last interval.
    offset = f'{(tests.time() % 60) + 5}s'

    resampled = reader.resample(**dict(df), rule=rule, offset=offset)
    return resampled.ffill(limit=1).iloc[-1]

def simple(reader, rule):
    name = '.'.join(reader.key.split('.')[1:])
    final = final_row(reader, rule).item()
    if isinstance(final, int):
        return [(name, f'{final:<6g}')]
    if isinstance(final, float):
        return [(name, f'{final:<6g}')]
    else:
        raise ValueError() 

def confidence(reader, rule):
    name = '.'.join(reader.key.split('.')[1:])
    final = final_row(reader, rule)
    return [(name, f'{final.μ:.2f}±{2*final.σ:.2f}')]


