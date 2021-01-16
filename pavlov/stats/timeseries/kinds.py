import numpy as np
import pandas as pd
from ... import tests, runs
from .factory import timeseries, arrays
from . import formatters, plotters

@timeseries()
def last(x, **kwargs):
    return x.resample(**kwargs).last().ffill()

@timeseries()
def max(x, **kwargs):
    return x.resample(**kwargs).max()

@timeseries(formatters.percent, plotters.Percent)
def max_percent(x, **kwargs):
    return x.resample(**kwargs).max()

@timeseries()
def mean(total, count=1, **kwargs):
    return total.resample(**kwargs).mean()/count.resample(**kwargs).mean()

@timeseries(formatters.percent, plotters.Percent)
def mean_percent(total, count=1, **kwargs):
    return total.resample(**kwargs).mean()/count.resample(**kwargs).mean()

@timeseries()
def std(x, **kwargs):
    return x.resample(**kwargs).std()

@timeseries()
def cumsum(total=1, **kwargs):
    return total.resample(**kwargs).sum().cumsum()

@timeseries()
def timeaverage(x, **kwargs):
    # TODO: To do this properly, I need to get individual per-device streams
    y = x.sort_index()
    dt = y.index.to_series().diff().dt.total_seconds()
    return (y*dt).resample(**kwargs).mean()/dt.resample(**kwargs).mean()

@timeseries(formatters.percent, plotters.Percent)
def duty(duration, **kwargs):
    sums = duration.resample(**kwargs).sum()
    periods = sums.index.to_series().diff().dt.total_seconds()
    return sums/periods

@timeseries()
def maxrate(duration, count=1, **kwargs):
    return count.resample(**kwargs).mean()/duration.resample(**kwargs).mean()

@timeseries()
def rate(count=1, **kwargs):
    counts = count.resample(**kwargs).sum()
    dt = pd.to_timedelta(counts.index.freq).total_seconds()
    dt = min(dt, (count.index[-1] - count.index[0]).total_seconds())
    return counts/dt

@timeseries()
def period(count=1, **kwargs):
    counts = count.resample(**kwargs).sum()
    dt = pd.to_timedelta(counts.index.freq).total_seconds()
    dt = min(dt, (count.index[-1] - count.index[0]).total_seconds())
    return dt/counts

@timeseries(formatters.confidence, plotters.Confidence)
def mean_std(μ, σ, **kwargs):
    μm = (μ/σ**2).resample(**kwargs).mean()/(1/σ**2).resample(**kwargs).mean()
    σm = 1/(1/σ**2).resample(**kwargs).mean()**.5
    return pd.concat({'μ': μm, 'σ': σm, 'μ-': μm - 2*σm, 'μ+': μm + 2*σm}, 1)

@timeseries(formatters.quantiles, plotters.Quantiles)
def quantiles(qs, **kwargs):
    averages = qs.resample(**kwargs).mean()
    averages.columns = [f'{100*q:.0f}' for q in np.linspace(0, 1, averages.shape[1])]
    return averages

@arrays(plotter=plotters.Line)
def line(xs, ys, **kwargs):
    return pd.Series(ys, xs).sort_index()

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
