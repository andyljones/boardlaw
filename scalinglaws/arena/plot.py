import pandas as pd
from . import database
import seaborn as sns
import matplotlib.pyplot as plt
from rebar import paths

def plot(df):
        if df.size <= 525:
            ax = sns.heatmap(df, cmap='RdBu', annot=True, vmin=0, vmax=1, annot_kws={'fontsize': 'large'}, cbar=False, square=True)
            return ax
        else:
            im = plt.imshow(df, cmap='RdBu', vmin=0, vmax=1)
            im.axes.set_xticks([])
            im.axes.set_yticks([])
            return im.axes

def plot_black(run_name):
    run_name = paths.resolve(run_name)
    df = (database.stored(run_name)
            .groupby(['black_name', 'white_name'])
            [['black_wins', 'white_wins']]
            .sum()
            .unstack())
    df = df.black_wins/(df.black_wins + df.white_wins)

    with plt.style.context('seaborn-poster'):
        ax = plot(df)
        ax.set_xlabel('white')
        ax.set_ylabel('black')
        ax.set_title('black win rate')

def plot_symmetric(run_name):
    run_name = paths.resolve(run_name)
    df = (database.stored(run_name)
            .groupby(['black_name', 'white_name'])
            [['black_wins', 'white_wins']]
            .sum()
            .unstack())
    games = df.black_wins + df.white_wins
    average = (df.black_wins + df.white_wins.T)/(games + games.T)

    with plt.style.context('seaborn-poster'):
        ax = plot(average)
        ax.set_xlabel(f'{run_name}')
        ax.set_ylabel('')
        ax.set_title(f'symmetrized win rate')

