import pandas as pd
from . import database
import seaborn as sns
import matplotlib.pyplot as plt

def plot(df):
    with plt.style.context('seaborn-poster'):
        ax = sns.heatmap(df, cmap='RdBu', annot=True, vmin=0, vmax=1, annot_kws={'fontsize': 'large'}, cbar=False, square=True)
        return ax

def plot_black():
    df = (database.stored()
            .assign(black_win=lambda df: df.black_reward == 1)
            .groupby(['black_name', 'white_name']).black_win.mean()
            .unstack())
    ax = plot(df)
    ax.set_xlabel('white')
    ax.set_ylabel('black')

def plot_symmetric():
    stats = (database.stored()
                .assign(
                    wins=lambda df: df.black_reward == 1,
                    games=lambda df: pd.Series(1, df.index))
                .groupby(['black_name', 'white_name'])[['wins', 'games']].sum()
                .astype(int)
                .unstack())

    average = (stats.wins + (stats.games - stats.wins).T)/(stats.games + stats.games.T)
    ax = plot(average)
    ax.set_xlabel('')
    ax.set_ylabel('')

