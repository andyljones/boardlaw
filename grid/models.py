import numpy as np
from . import data
import pandas as pd
import statsmodels.formula.api as smf
from sklearn import metrics

def linear_model(aug):
    model = smf.ols('rel_elo ~ boardsize + np.log2(width) + np.log2(depth) + 1', aug).fit()
    return model.fittedvalues

def run():
    aug = data.augmented()

    pred = linear_model(aug)
    metrics.r2_score(aug.rel_elo, pred)
