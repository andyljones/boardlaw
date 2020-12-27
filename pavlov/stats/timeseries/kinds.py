import pandas as pd
from ... import tests, runs
from .template import timeseries

@timeseries
def last(x, **kwargs):
    return x.resample(**kwargs).last().ffill()

@timeseries
def max(x, **kwargs):
    return x.resample(**kwargs).max()

@timeseries
def mean(total, count=1, **kwargs):
    return total.resample(**kwargs).mean()/count.resample(**kwargs).mean()

@timeseries
def std(x, **kwargs):
    return x.resample(**kwargs).std()

@timeseries
def cumsum(total=1, **kwargs):
    return total.resample(**kwargs).sum().cumsum()

@timeseries
def timeaverage(x, **kwargs):
    # TODO: To do this properly, I need to get individual per-device streams
    y = x.sort_index()
    dt = y.index.to_series().diff().dt.total_seconds()
    return (y*dt).resample(**kwargs).mean()/dt.resample(**kwargs).mean()

@timeseries
def duty(duration, **kwargs):
    sums = duration.resample(**kwargs).sum()
    periods = sums.index.to_series().diff().dt.total_seconds()
    return sums/periods

@timeseries
def maxrate(duration, count=1, **kwargs):
    return count.resample(**kwargs).mean()/duration.resample(**kwargs).mean()

@timeseries
def rate(count=1, **kwargs):
    counts = count.resample(**kwargs).sum()
    dt = pd.to_timedelta(counts.index.freq).total_seconds()
    dt = min(dt, (count.index[-1] - count.index[0]).total_seconds())
    return counts/dt

@timeseries
def period(count=1, **kwargs):
    counts = count.resample(**kwargs).sum()
    dt = pd.to_timedelta(counts.index.freq).total_seconds()
    dt = min(dt, (count.index[-1] - count.index[0]).total_seconds())
    return dt/counts

#TODO:
# * log_cumsum
# * mean_std
# * dist

@tests.mock_time
@tests.mock_dir
def test_mean():
    from .. import to_run

    run = runs.new_run()
    with to_run(run):
        tests.set_time(0)
        mean('test', 4, 2)
        tests.set_time(1)
        mean('test', 8, 2)

    reader = mean.Reader(run, 'test')
    final = reader.final('60s')
    assert final == 3.

    [label, value] = reader.format('60s')
    assert label == 'test'
    assert value == '3'
