import numpy as np
import torch
from torch import nn
import pandas as pd
import scipy as sp
from tqdm.auto import tqdm
from boardlaw import sql, elos
import aljpy
from pavlov import stats, runs
import pandas as pd


@aljpy.autocache()
def _trial_elos(boardsize, counter):
    trials = sql.trial_query(boardsize, 'bee/%')
    ws, gs = elos.symmetrize(trials)
    return elos.solve(ws, gs)

def trial_elos(boardsize):
    counter = sql.file_change_counter()
    return _trial_elos(boardsize, counter)

def load():
    ags = sql.agent_query()

    es = []
    for b in tqdm(ags.boardsize.unique()):
        es.append(trial_elos(b))
    es = pd.concat(es)

    return ags.join(es, how='inner')

def with_times(ags):
    rates = {}
    for r in ags.run.unique():
        arr = stats.array(r, 'count.samples')
        s, t = arr['total'], arr['_time']
        rates[r] = 1e6*(s.sum() - s[0])/(t[-1] - t[0]).astype(float)
    rates = pd.Series(rates, name='sample_rate')

    aug = pd.merge(ags, rates, left_on='run', right_index=True)
    aug['train_time'] = aug.samples/aug.sample_rate
    return aug

def interp_frontier(g, x='train_flops', y='elo', group='run'):
    xl, xr = g[x].pipe(np.log10).min(), g[x].pipe(np.log10).max()
    xs = np.linspace(xl, xr, 101)
    ys = {}
    for run, gg in g.sort_values(x).groupby(group):
        xp = gg[x].pipe(np.log10).values
        yp = gg[y].values
        ys[run] = np.interp(xs, xp, yp, np.nan, np.nan)
    ys = pd.DataFrame(ys, index=10**xs)

    return ys.max(1).rename_axis(index=x).rename(y)

class Changepoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.lower = nn.Parameter(torch.as_tensor([-1.5, 3.]))
        self.linear = nn.Parameter(torch.as_tensor([2., -2, -16]))
        
    def forward(self, X):
        X = torch.cat([X, torch.ones_like(X[:, :1])], -1)
        lower = X[:, 1:] @ self.lower
        linear = X @ self.linear
        return torch.maximum(linear, lower).clamp(None, 0)

class Sigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.as_tensor([1/16., 0.]))
        self.height = nn.Parameter(torch.as_tensor(1.3))
        self.center = nn.Parameter(torch.as_tensor([.66, 9.]))
        
    def forward(self, X):
        X = torch.cat([X, torch.ones_like(X[:, :1])], -1)
        hscale = X[:, 1:] @ self.scale
        vscale = hscale * self.height
        center = X[:, 1:] @ self.center
        return vscale*(torch.sigmoid((X[:, 0] - center)/hscale) - 1)

def model_inputs(df):
    return torch.stack([
        torch.as_tensor(df.train_flops.values).log10().float(),
        torch.as_tensor(df.boardsize.values).float(),], -1)

def fit_model(df):
    X = model_inputs(df)
    y = torch.as_tensor(df.elo.values)

    model = Sigmoid()
    optim = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe', max_iter=100)

    def closure():
        yhat = model(X)
        loss = (y - yhat).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        return loss
        
    optim.step(closure)

    return model

def apply_model(model, df):
    X = model_inputs(df)
    return pd.Series(model(X).detach().cpu().numpy(), df.index)
    
def perfect_play(model, target=-100):
    perfect = {}
    for b in range(3, 10):
        f = lambda x: 400/np.log(10)*model(torch.as_tensor([[x, b]])).detach().numpy().squeeze() - target
        perfect[b] = sp.optimize.bisect(f, 1, 18)
    return pd.Series(perfect, name='perfect')

def modelled_elos(ags):
    df = (ags.query('test_nodes == 64')
            .groupby('boardsize')
            .apply(interp_frontier, 'train_flops')
            .reset_index()) 
    
    model = fit_model(df)
    df['elohat'] = apply_model(model, df)
    return df, model

def residual_vars(ags):
    df = (ags.query('test_nodes == 64')
        .groupby('boardsize')
        .apply(interp_frontier, 'train_flops')
        .reset_index()) 

    yhats = {}
    for b in range(4, 10):
        model = fit_model(df[df.boardsize <= b])
        yhats[b] = apply_model(model, df[df.boardsize >= b])
    yhats = pd.concat(yhats, 1)

    num = yhats.sub(df.elo, 0).pow(2).groupby(df.boardsize).mean()
    denom = df.elo.pow(2).groupby(df.boardsize).mean()
    resid_var = (num/denom).stack().reset_index()
    resid_var.columns = ['predicted', 'seen', 'rv']

    perfect = perfect_play(model)
    resid_var = (resid_var
        .join(perfect.rename('predicted_perfect'), on='predicted')
        .join(perfect.rename('seen_perfect'), on='seen')
        .assign(ratio=lambda df: 10**(df.seen_perfect - df.predicted_perfect)))

    return resid_var
 