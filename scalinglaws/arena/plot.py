import pandas as pd
from . import database
import seaborn as sns
import matplotlib.pyplot as plt
from rebar import paths

def plot(df, vmin=0, vmax=1):
    if df.size <= 525:
        ax = sns.heatmap(df, cmap='RdBu', annot=True, vmin=vmin, vmax=vmax, annot_kws={'fontsize': 'large'}, cbar=False, square=True)
        return ax
    else:
        im = plt.imshow(df, cmap='RdBu', vmin=vmin, vmax=vmax)
        im.axes.set_xticks([])
        im.axes.set_yticks([])
        return im.axes

def games(run_name):
    df = database.games(run_name)
    with plt.style.context('seaborn-poster'):
        ax = plot(df, vmax=df.max().max())

def black(*args, **kwargs):
    df = database.wins(*args, **kwargs)
    with plt.style.context('seaborn-poster'):
        ax = plot(df)
        ax.set_xlabel('white')
        ax.set_ylabel('black')
        ax.set_title('black win rate')

def symmetric(*args, **kwargs):
    df = database.symmetric_winrate(*args, **kwargs)
    with plt.style.context('seaborn-poster'):
        ax = plot(df)
        ax.set_ylabel('')
        ax.set_title(f'symmetrized win rate')

