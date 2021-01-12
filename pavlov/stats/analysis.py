import numpy as np
import pandas as pd
from . import registry
from .. import runs
from logging import getLogger

log = getLogger(__name__)

def array(run, channel):
    return registry.reader(run, channel).array()

def pandas(run, channel, field=None, rule='60s', **kwargs):
    r = registry.reader(run, channel)
    if not r.ready():
        raise ValueError(f'Reader for "{run}" "{channel}" is not ready')
    df = r.resample(rule, **kwargs)
    if field is not None:
        df = df[field]
    return df

def compare(rs, *args, fill=False, **kwargs):
    rs = [rs] if isinstance(rs, str) else rs
    ns = [n for r in rs for n in runs.resolutions(r)]
    df = {}
    for n in ns:
        try:
            df[n] = pandas(n, *args, **kwargs)
        except OSError:
            log.info(f'Couldn\'t find data for "{n}"')

    return pd.concat(df, 1)

def plot(*args, fill=False, skip=None, head=None, **kwargs):
    df = compare(*args, **kwargs)
    if fill:
        df = df.ffill().where(df.bfill().notnull())
    ax = df.iloc[skip:head].plot()
    ax.grid(True)
    return ax

def purge(minlen=900, cutoff=300):
    raise ValueError('Make sure this doesn\'t delete snapshot-only runs!')
    from tqdm.auto import tqdm
    for r in tqdm(runs.runs()):
        try:
            start = pd.to_datetime(runs.info(r)['_created'])
            end = np.datetime64(start.tz_localize(None))
            #TODO: Just load the first and last line of the file
            for _, reader in registry.StatsReaders(r).items():
                end = max(end, reader.array()['_time'].max())
            end = pd.to_datetime(end).tz_localize('UTC')
            length = end - start
            cut = pd.Timestamp.now('UTC') - pd.to_timedelta(cutoff, 's')
            if length.total_seconds() < minlen and end < cut:
                print(f'Deleting {length.total_seconds():.0f}s run "{r}"')
                runs.delete(r)
        except Exception as e:
            print(f'Checking "{r}" failed with error "{e}"')