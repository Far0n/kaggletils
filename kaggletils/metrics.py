# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y, yhat):
    return mean_squared_error(y, yhat) ** 0.5


def gini(y, yhat, cmpcol=0, sortcol=1):
    assert (len(y) == len(yhat))

    all = np.asarray(np.c_[y, yhat, np.arange(len(y))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]

    total_loss = all[:, 0].sum()
    sum = all[:, 0].cumsum().sum() / total_loss

    sum -= (len(y) + 1) / 2.
    return sum / len(y)


def gini_normalized(y, y_hat):
    return gini(y, y_hat) / gini(y, y)


def logloss(y, yhat):
    yhat = max(min(yhat, 1. - 10e-15), 10e-15)
    return -np.log(yhat) if y == 1. else -np.log(1. - yhat)


def rmspe(y, yhat):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=5, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def ndgc_k(y, yhat, k=5):
    top = []
    for i in range(yhat.shape[0]):
        top.append(np.argsort(yhat[i])[::-1][:k])
    mat = np.reshape(np.repeat(y, np.shape(top)[1]) == np.array(top).ravel(), np.array(top).shape).astype(int)
    return np.mean(np.sum(mat / np.log2(np.arange(2, mat.shape[1] + 2)), axis=1))


def ndgc5(y, yhat):
    top = []
    for i in range(yhat.shape[0]):
        top.append(np.argsort(yhat[i])[::-1][:5])
    mat = np.reshape(np.repeat(y, np.shape(top)[1]) == np.array(top).ravel(), np.array(top).shape).astype(int)
    return np.mean(np.sum(mat / np.log2(np.arange(2, mat.shape[1] + 2)), axis=1))


def ndgc10(y, yhat):
    top = []
    for i in range(yhat.shape[0]):
        top.append(np.argsort(yhat[i])[::-1][:10])
    mat = np.reshape(np.repeat(y, np.shape(top)[1]) == np.array(top).ravel(), np.array(top).shape).astype(int)
    return np.mean(np.sum(mat / np.log2(np.arange(2, mat.shape[1] + 2)), axis=1))


def ap_at_k(y, yhat, k=5):
    if len(yhat) > k:
        yhat = yhat[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(yhat):
        if p in y and p not in yhat[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not y:
        return 0.0

    return score / min(len(y), k)


def map_at_k(y, yhat, k=5):
    return np.mean([ap_at_k(a, p, k) for a, p in zip(y, yhat)])


def map5(y, yhat):
    return map_at_k(y, yhat, 5)


def map10(y, yhat):
    return map_at_k(y, yhat, 10)