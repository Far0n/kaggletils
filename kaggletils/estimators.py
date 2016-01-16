# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""

from __future__ import division

import numpy as np
import pandas as pd
from numpy.random.mtrand import normal, multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array


class LikelihoodEstimatorUnivariate(BaseEstimator):
    def __init__(self, seed=0, alpha=0, noise=0, leave_one_out=False):
        self.alpha = alpha
        self.noise = noise
        self.seed = seed
        self.leave_one_out = leave_one_out
        self.nclass = None
        self.classes = None
        self.class_priors = None
        self.likelihoods = None
        self.x_likelihoods = None

    def fit(self, x, y):
        np.random.seed(self.seed)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if x.shape[1] != 1:
            raise ValueError('x must be one dimensional.')

        x, y = check_X_y(x, y)

        self.classes = np.unique(y)
        self.nclass = self.classes.shape[0]

        ctab = pd.crosstab(x[:, 0], y).reset_index()
        xcol = ctab.columns[0]
        ycols = list(ctab.columns[1:])

        xtab = pd.DataFrame(x).rename(columns={0: xcol})
        xtab = xtab.merge(ctab, how='left', on=xcol)

        self.class_priors = xtab[ycols].div(xtab[ycols].sum(axis=1), axis=0).mean().values

        if self.leave_one_out:
            xtab[ycols] -= pd.get_dummies(y)

        xtab[ycols] = xtab[ycols].add(self.class_priors * self.alpha). \
            div(xtab[ycols].sum(axis=1) + self.alpha + 1E-15, axis=0)
        if self.noise > 0:
            xtab[ycols] = np.abs(xtab[ycols] + normal(0, scale=self.noise, size=xtab[ycols].shape))
            xtab[ycols] = xtab[ycols].div(xtab[ycols].sum(axis=1), axis=0)
        self.x_likelihoods = xtab[ycols].values

        xtab_agg = xtab.groupby(xcol, as_index=False)[ycols].agg(['mean', 'std']).fillna(0)
        xtab_agg.columns = xtab_agg.columns.get_level_values(1)

        self.likelihoods = xtab_agg.T.ix['mean'].reset_index(drop=True).T
        # self.likelihoods = xtab_agg.T.ix['mean'].reset_index(drop=True).to_dict('list')
        # self.likelihoods_cov = xtab_agg.T.ix['std'].reset_index(drop=True).to_dict('list')
        # self.likelihoods_cov = dict((k, np.diag(v)) for k, v in self.likelihoods_cov.items())

        return self

    def _calc_likelihood(self, x):
        return (x + self.class_priors * self.alpha) / (x.sum() + self.alpha)

    def _get_likelihood(self, x, noise):
        mean = self.likelihoods.get(x[0], self.class_priors)
        cov = self.likelihoods_cov.get(x[0], np.diag(np.zeros((self.nclass,))))
        if noise:
            if isinstance(noise, float):
                cov = np.diag(np.ones((self.nclass,)) * noise)
            lh = np.abs(multivariate_normal(mean, cov))
            return lh / lh.sum()
        else:
            return mean

    def predict(self, x, noise=False, normalize=False):
        if normalize:
            return np.average(self.predict_proba(x, noise), axis=1, weights=self.classes)
        else:
            return np.dot(self.predict_proba(x, noise), self.classes)

    def predict_proba(self, x, noise=False):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if x.shape[1] != 1:
            raise ValueError('x must be one dimensional.')

        xx = pd.DataFrame(x, columns=['x']).merge(self.likelihoods, how='left', left_on='x', right_index=True)
        xx.drop('x', axis=1, inplace=True)
        xx.loc[xx.isnull().any(axis=1) | (xx == 0).all(axis=1), :] = self.class_priors

        if noise:
            np.random.seed(self.seed)
            _noise = noise if isinstance(noise, float) else self.noise
            if _noise > 1E-12:
                xx = np.abs(xx + normal(0, scale=_noise, size=xx.shape))
                xx = xx.div(xx.sum(axis=1), axis=0)

        # return np.apply_along_axis(self._get_likelihood, 1, x, noise)
        return xx.values


class LikelihoodEstimator(BaseEstimator):
    def __init__(self, seed=0, alpha=0, noise=0, leave_one_out=False):
        self.alpha = alpha
        self.noise = noise
        self.seed = seed
        self.leave_one_out = leave_one_out
        self.nclass = None
        self.classes = None
        self.class_priors = None
        self.likelihoods = None
        self.x_likelihoods = None

    def fit(self, x, y):
        np.random.seed(self.seed)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        x, y = check_X_y(x, y)

        self.classes = np.unique(y)
        self.nclass = self.classes.shape[0]

        ctab = pd.crosstab(y, list(x.T)).T.reset_index()

        xdim = x.shape[1]
        xcols = list(ctab.columns[:xdim])
        ycols = list(ctab.columns[xdim:])

        xtab = pd.DataFrame(x, columns=xcols)
        xtab = xtab.merge(ctab, how='left', on=xcols)

        self.class_priors = xtab[ycols].div(xtab[ycols].sum(axis=1), axis=0).mean().values

        if self.leave_one_out:
            xtab[ycols] -= pd.get_dummies(y)

        xtab[ycols] = xtab[ycols].add(self.class_priors * self.alpha). \
            div(xtab[ycols].sum(axis=1) + self.alpha + 1E-15, axis=0)
        if self.noise > 0:
            xtab[ycols] = np.abs(xtab[ycols] + normal(0, scale=self.noise, size=xtab[ycols].shape))
            xtab[ycols] = xtab[ycols].div(xtab[ycols].sum(axis=1), axis=0)
        self.x_likelihoods = xtab[ycols].values

        xtab_agg = xtab.groupby(xcols, as_index=False)[ycols].agg(['mean']).fillna(0)
        xtab_agg.columns = xtab_agg.columns.get_level_values(1)

        self.likelihoods = xtab_agg.T.ix['mean'].reset_index(drop=True).T.reset_index()
        # self.likelihoods = xtab_agg.T.ix['mean'].reset_index(drop=True).to_dict('list')
        # self.likelihoods_cov = xtab_agg.T.ix['std'].reset_index(drop=True).to_dict('list')
        # self.likelihoods_cov = dict((k, np.diag(v)) for k, v in self.likelihoods_cov.items())

        return self

    def _calc_likelihood(self, x):
        return (x + self.class_priors * self.alpha) / (x.sum() + self.alpha)

    def _get_likelihood(self, x, noise):
        mean = self.likelihoods.get(x[0], self.class_priors)
        cov = self.likelihoods_cov.get(x[0], np.diag(np.zeros((self.nclass,))))
        if noise:
            if isinstance(noise, float):
                cov = np.diag(np.ones((self.nclass,)) * noise)
            lh = np.abs(multivariate_normal(mean, cov))
            return lh / lh.sum()
        else:
            return mean

    def predict(self, x, noise=False, normalize=False):
        if normalize:
            return np.average(self.predict_proba(x, noise), axis=1, weights=self.classes)
        else:
            return np.dot(self.predict_proba(x, noise), self.classes)

    def predict_proba(self, x, noise=False):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        x = check_array(x)

        xx = pd.DataFrame(x, columns=self.likelihoods.columns[:-self.nclass])
        xx = xx.merge(self.likelihoods, how='left')
        xx.drop(xx.columns[:-self.nclass], axis=1, inplace=True)
        xx.loc[xx.isnull().any(axis=1) | (xx == 0).all(axis=1), :] = self.class_priors

        if noise:
            np.random.seed(self.seed)
            _noise = noise if isinstance(noise, float) else self.noise
            if _noise > 1E-12:
                xx = np.abs(xx + normal(0, scale=_noise, size=xx.shape))
                xx = xx.div(xx.sum(axis=1), axis=0)

        # return np.apply_along_axis(self._get_likelihood, 1, x, noise)
        return xx.values
