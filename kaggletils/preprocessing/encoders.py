# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""

from __future__ import division

from collections import Counter

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from statsmodels.distributions import ECDF

from ..estimators import LikelihoodEstimator
from ..utils.data import is_numpy


class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=0, nan_value=-1, copy=True):
        self.min_count = min_count
        self.nan_value = nan_value
        self.copy = copy
        self.counts = {}

    def fit(self, x):
        self.counts = {}
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        is_np = is_numpy(x)

        for i in range(ncols):
            if is_np:
                cnt = dict(Counter(x[:, i]))
            else:
                cnt = x.iloc[:, i].value_counts().to_dict()
            if self.min_count > 0:
                cnt = dict((k, self.nan_value if v < self.min_count else v) for k, v in cnt.items())
            self.counts.update({i: cnt})
        return self

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        if self.copy:
            x = x.copy()
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        is_np = is_numpy(x)

        for i in range(ncols):
            cnt = self.counts[i]
            if is_np:
                k, v = np.array(list(zip(*sorted(cnt.items()))))
                ix = np.digitize(x[:, i], k, right=True)
                x[:, i] = v[ix]
            else:
                x.iloc[:, i].replace(cnt, inplace=True)
        return x


class LikelihoodEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, seed=0, alpha=0, leave_one_out=False, noise=0):
        self.alpha = alpha
        self.noise = noise
        self.seed = seed
        self.leave_one_out = leave_one_out
        self.nclass = None
        self.estimators = []

    def fit(self, x, y):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        if not is_numpy(x):
            x = np.array(x)

        self.nclass = np.unique(y).shape[0]

        for i in range(ncols):
            self.estimators.append(LikelihoodEstimator(**self.get_params()).fit(x[:, i], y))
        return self

    @staticmethod
    def owen_zhang(x_train, y_train, x_test, seed=0, alpha=0, noise=0.01):
        """
        Owen Zhang's leave-one-out + noise likelihood encoding

        "Winning data science competitions"
        http://de.slideshare.net/ShangxuanZhang/winning-data-science-competitions-presented-by-owen-zhang
        """
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)
        ncols = x_train.shape[1]
        nclass = np.unique(y_train).shape[0]
        if not is_numpy(x_train):
            x_train = np.array(x_train)
            x_test = np.array(x_test)

        xx_train = None
        xx_test = None

        for i in range(ncols):
            le_train = LikelihoodEstimator(noise=noise, alpha=alpha, leave_one_out=True, seed=seed). \
                fit(x_train[:, i], y_train)
            le_test = LikelihoodEstimator(noise=0, alpha=alpha, leave_one_out=False, seed=seed). \
                fit(x_train[:, i], y_train)
            lh_train = le_train.x_likelihoods.copy()
            lh_test = le_test.predict_proba(x_test[:, i])

            if nclass <= 2:
                lh_train = lh_train.T[1].reshape(-1, 1)
                lh_test = lh_test.T[1].reshape(-1, 1)

            xx_train = np.hstack((lh_train,)) if xx_train is None else np.hstack((xx_train, lh_train))
            xx_test = np.hstack((lh_test,)) if xx_test is None else np.hstack((xx_test, lh_test))

        return xx_train, xx_test

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        if not is_numpy(x):
            x = np.array(x)

        likelihoods = None

        for i in range(ncols):
            lh = self.estimators[i].predict(x[:, i], noise=True).reshape(-1, 1)
            # lh = self.estimators[i].predict_proba(x[:, i])
            # if self.nclass <= 2:
            #     lh = lh.T[1].reshape(-1, 1)
            likelihoods = np.hstack((lh,)) if likelihoods is None else np.hstack((likelihoods, lh))
        return likelihoods


class PercentileEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, apply_ppf=False, copy=True):
        self.ppf = lambda x: norm.ppf(x * .998 + .001) if apply_ppf else x
        self.copy = copy
        self.ecdfs = {}

    def fit(self, x):
        self.ecdfs = {}
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        is_np = is_numpy(x)

        for i in range(ncols):
            self.ecdfs.update({i: ECDF(x[:, i] if is_np else x.iloc[:, i].values)})
        return self

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        if self.copy:
            x = x.copy()
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        is_np = is_numpy(x)

        for i in range(ncols):
            ecdf = self.ecdfs[i]
            if is_np:
                x[:, i] = self.ppf(ecdf(x[:, i]))
            else:
                x.iloc[:, i] = self.ppf(ecdf(x.iloc[:, i]))
        return x


class InfrequentValueEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10, value=-1, copy=True):
        self.threshold = threshold
        self.value = value
        self.copy = copy
        self.new_values = {}

    def fit(self, x):
        self.new_values = {}
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        is_np = is_numpy(x)

        for i in range(ncols):
            if is_np:
                val = dict(Counter(x[:, i]))
            else:
                val = x.iloc[:, i].value_counts().to_dict()
            val = dict((k, self.value if v < self.threshold else k) for k, v in val.items())
            self.new_values.update({i: val})
        return self

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        if self.copy:
            x = x.copy()
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        is_np = is_numpy(x)

        for i in range(ncols):
            val = self.new_values[i]
            if is_np:
                k, v = np.array(list(zip(*sorted(val.items()))))
                ix = np.digitize(x[:, i], k, right=True)
                x[:, i] = v[ix]
            else:
                x.iloc[:, i].replace(val, inplace=True)
        return x


class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=0, first_category=1, copy=True):
        self.min_count = min_count
        self.first_category = first_category
        self.copy = copy
        self.encoders = {}
        self.ive = None

    def fit(self, x):
        self.encoders = {}
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        is_np = is_numpy(x)

        if self.min_count > 0:
            self.ive = InfrequentValueEncoder(threshold=self.min_count, value=np.finfo(float).min)
            x = self.ive.fit_transform(x)

        for i in range(ncols):
            if is_np:
                enc = LabelEncoder().fit(x[:, i])
            else:
                enc = LabelEncoder().fit(x.iloc[:, i])
            self.encoders.update({i: enc})
        return self

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        if self.copy:
            x = x.copy()
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        is_np = is_numpy(x)

        if self.ive is not None:
            x = self.ive.transform(x)

        for i in range(ncols):
            enc = self.encoders[i]
            if is_np:
                x[:, i] = enc.transform(x[:, i]) + self.first_category
            else:
                x.iloc[:, i] = enc.transform(x.iloc[:, i]) + self.first_category
        return x


class DummyEncoder(BaseEstimator, TransformerMixin):
    pass