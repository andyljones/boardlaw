import pandas as pd
from . import database
import seaborn as sns
import matplotlib.pyplot as plt

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
    df = (database.stored(run_name)
            .assign(black_win=lambda df: df.black_reward == 1)
            .groupby(['black_name', 'white_name']).black_win.mean()
            .unstack())

    with plt.style.context('seaborn-poster'):
        ax = plot(df)
        ax.set_xlabel('white')
        ax.set_ylabel('black')
        ax.set_title('black win rate')

def plot_symmetric(run_name):
    stats = (database.stored(run_name)
                .assign(
                    wins=lambda df: df.black_reward == 1,
                    games=lambda df: pd.Series(1, df.index))
                .groupby(['black_name', 'white_name'])[['wins', 'games']].sum()
                .astype(int)
                .unstack())
    average = (stats.wins + (stats.games - stats.wins).T)/(stats.games + stats.games.T)

    with plt.style.context('seaborn-poster'):
        ax = plot(average)
        ax.set_xlabel(f'{run_name}')
        ax.set_ylabel('')
        ax.set_title(f'symmetrized win rate')

