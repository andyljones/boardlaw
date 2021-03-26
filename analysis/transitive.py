import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from boardlaw.arena import common
from IPython.display import clear_output

def _elo(r):
    return 400/np.log(10)*(np.log(1 - r) - np.log(r))

def elo_range(m, n, q=.1):
    post = sp.stats.beta(m+1, n+1)
    return _elo(post.ppf(.5)), abs(_elo(post.ppf(q)) - _elo(post.ppf(1-q)))

def run():
    ref = ('2021-02-20 23-35-25 simple-market', 20)
    test = ('2021-02-20 22-17-16 bulky-frosts', 9)
    worlds = common.worlds(ref[0], 16*1024, device='cuda')
    ref = common.agent(*ref, device='cuda')
    test = common.agent(*test, device='cuda')

    df = pd.DataFrame(0, index=['ref', 'test'], columns=['black', 'white'])
    while True:
        results = common.evaluate(worlds, {'ref': ref, 'test': test})
        ws = pd.concat([pd.Series(r.wins, r.names) for r in results], 1)
        ws.columns = ['black', 'white']
        df += ws

        wins = df.sum(1)
        centre, gap = elo_range(wins.ref, wins.test)
        
        clear_output(wait=True)
        print(df)
        print(f'{centre:.0f}±{gap/2:.0f}')

        if gap < 50:
            break
