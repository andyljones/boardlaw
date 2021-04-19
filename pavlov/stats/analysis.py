import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from . import registry
from .. import runs, files
from logging import getLogger

log = getLogger(__name__)

def array(run, channel):
    return registry.reader(run, channel).array()

def pandas(run, channel, field=None, rule='60s', **kwargs):
    r = registry.reader(run, channel)
    if not r.ready():
        raise ValueError(f'Reader for "{run}" "{channel}" is not ready')

    if rule is None:
        df = r.pandas()
    else:
        df = r.resample(rule, **kwargs)
    
    if field is not None:
        df = df[field]
    return df

def compare(rs, channel, *args, query='', **kwargs):
    if not isinstance(channel, str):
        return pd.concat({c: compare(rs, c, *args, query=query, **kwargs) for c in channel}, 1)
    rs = [rs] if isinstance(rs, str) else rs
    ns = [n for r in rs for n in (runs.pandas(r).query(query) if query else runs.pandas(r)).index]
    df = {}
    for n in ns:
        try:
            df[n] = pandas(n, channel, *args, **kwargs)
        except OSError:
            log.info(f'Couldn\'t find data for "{n}"')

    return pd.concat(df, 1)

def plot(*args, ffill=False, skip=None, head=None, **kwargs):
    df = compare(*args, **kwargs)
    desc = runs.pandas().loc[df.columns, 'description'].fillna('')
    df.columns = [f'{c}: {desc[c]}' for c in df.columns]
    if ffill:
        df = df.ffill().where(df.bfill().notnull())
    ax = df.iloc[skip:head].plot()
    ax.grid(True)
    return ax

def periodic(*args, period=900, **kwargs):
    from IPython import display

    epoch = pd.Timestamp('2020-1-1') 
    last = 0
    ax = None
    while True:
        now = pd.Timestamp.now()
        new = int((now - epoch).total_seconds())//period
        if new > last:
            display.clear_output(wait=True)
            if ax is not None:
                plt.close(ax.figure)
            ax = plot(*args, **kwargs)
            ax.set_title(f'{now:%Y-%m-%d %H:%M:%S}')
            display.display(ax.figure)
            last = new
        else:
            time.sleep(1)

def purge(minlen=300, cutoff=900):
    from tqdm.auto import tqdm
    for r in tqdm(runs.runs()):
        try:
            start = pd.to_datetime(runs.info(r)['_created'])
            end = start
            for f in files.files(r):
                mtime = pd.Timestamp(files.path(r, f).lstat().st_mtime, unit='s', tz='UTC')
                end = max(end, mtime)
            length = end - start
            cut = pd.Timestamp.now('UTC') - pd.to_timedelta(cutoff, 's')
            if length.total_seconds() < minlen and end < cut:
                print(f'Deleting {length.total_seconds():.0f}s run "{r}"')
                runs.delete(r)
        except Exception as e:
            raise
            print(f'Checking "{r}" failed with error "{e}"')