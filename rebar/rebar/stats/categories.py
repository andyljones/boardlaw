from rebar import dotdict
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
import logging
from . import formatters, plotters

log = logging.getLogger(__name__)

CATEGORIES = {}
def category(formatter=formatters.single, plotter=plotters.single):
    def add(M):
        CATEGORIES[M.__name__.lower()] = dotdict.dotdict(
            resampler=M,
            formatter=formatter,
            plotter=plotter)
        return M
    return add

@category()
def last(x):
    def resample(**kwargs):
        return x.resample(**kwargs).last().ffill()
    return resample

@category()
def max(x):
    def resample(**kwargs):
        return x.resample(**kwargs).max()
    return resample

@category()
def mean(total, count=1):
    def resample(**kwargs):
        return total.resample(**kwargs).mean()/count.resample(**kwargs).mean()
    return resample

@category(formatters.confidence, plotters.confidence)
def mean_std(μ, σ):
    def resample(**kwargs):
        μm = (μ/σ**2).resample(**kwargs).mean()/(1/σ**2).resample(**kwargs).mean()
        σm = 1/(1/σ**2).resample(**kwargs).mean()**.5
        return pd.concat({'μ': μm, 'σ': σm, 'μ-': μm - 2*σm, 'μ+': μm + 2*σm}, 1)
    return resample

@category()
def std(x):
    def resample(**kwargs):
        return x.resample(**kwargs).std()
    return resample

@category(plotter=plotters.log_single)
def cumsum(total=1):
    def resample(**kwargs):
        return total.resample(**kwargs).sum().cumsum()
    return resample

@category(plotter=plotters.log_single)
def log_cumsum(total=1):
    def resample(**kwargs):
        return total.resample(**kwargs).sum().cumsum()
    return resample

@category()
def timeaverage(x):
    def resample(**kwargs):
        # TODO: To do this properly, I need to get individual per-device streams
        y = x.sort_index()
        dt = y.index.to_series().diff().dt.total_seconds()
        return (y*dt).resample(**kwargs).mean()/dt.resample(**kwargs).mean()
    return resample

@category()
def duty(duration):
    def resample(**kwargs):
        sums = duration.resample(**kwargs).sum()
        periods = sums.index.to_series().diff().dt.total_seconds()
        return sums/periods
    return resample

@category()
def maxrate(duration, count=1):
    def resample(**kwargs):
        return count.resample(**kwargs).mean()/duration.resample(**kwargs).mean()
    return resample

@category()
def rate(count=1):
    def resample(**kwargs):
        counts = count.resample(**kwargs).sum()
        dt = pd.to_timedelta(counts.index.freq).total_seconds()
        dt = min(dt, (count.index[-1] - count.index[0]).total_seconds())
        return counts/dt
    return resample

@category()
def period(count=1):
    def resample(**kwargs):
        counts = count.resample(**kwargs).sum()
        dt = pd.to_timedelta(counts.index.freq).total_seconds()
        dt = min(dt, (count.index[-1] - count.index[0]).total_seconds())
        return dt/counts
    return resample

@category()
def dist(samples, size=10000):
    return samples

@category()
def noisescale(S, G2, B):
    def resample(**kwargs):
        return S.resample(**kwargs).mean()/G2.resample(**kwargs).mean()
    return resample