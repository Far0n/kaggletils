# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""

import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from . import is_numpy


def logit(x):
    x = np.clip(x, 1E-15, 1.0 - 1E-15)
    return np.log(x / (1.0 - x))


def logistic(x):
    return 1. / (1.0 + np.exp(-x))


def additive_smoothing(x, priors, alpha=1):
    return (x + priors * alpha) / (x.sum() + alpha)


def anscombe(x):
    return 2. * np.sqrt(x + 3. / 8.)


def entropy(x, y=np.empty):
    if isinstance(x, str):
        x = np.array(list(x))
    if isinstance(y, str):
        y = np.array(list(y))
    if not is_numpy(x):
        x = np.array(x)
    if not is_numpy(y):
        y = np.array(y)

    ctab = pd.crosstab(1, [x, y], margins=False).T.reset_index(drop=True)
    # ctab = ctab.div(ctab.sum(axis=0), axis=1).apply(lambda p: -p * np.log2(p))
    # return ctab.sum().values[0]
    return scipy.stats.entropy(ctab.values, base=2)[0]


def conditional_entropy(x, y, bins=1):
    if isinstance(x, str) or bins <= 1:
        return entropy(x, y) - entropy(y)
    return entropy(x) - mutual_information(x, y, bins)


def mutual_information(x, y, bins=1):
    if isinstance(x, str) or bins <= 1:
        return entropy(x) + entropy(y) - entropy(x, y)
    return mutual_info_score(None, None, contingency=np.histogram2d(x, y, bins)[0])
