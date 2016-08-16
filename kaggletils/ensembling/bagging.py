# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""

from datetime import datetime

import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics import log_loss


class Bagger(object):
    def __init__(self, clf, clf_params=None, nbags=10, seed=0, regression=False,
                 subsample=1., bootstrap=False, shuffle=False, metric=log_loss, verbose=True):
        self.clf = clf
        self.clf_params = clf_params if clf_params is not None else {}
        self.seed = seed
        self.regression = regression
        self.nbags = nbags
        self.subsample = subsample
        self.bootstrap = bootstrap
        self.shuffle = shuffle
        self.metric = metric
        self.verbose = verbose

    def bag(self, x_train, y_train, x_test, x_probe=None, y_probe=None, sample_weights=None):
        ts = datetime.now()
        self.ntrain = x_train.shape[0]
        self.ntest = x_test.shape[0]
        self.nprobe = 0 if x_probe is None else x_probe.shape[0]
        self.sample_weights = sample_weights
        self.nclass = 1 if self.regression else np.unique(y_train).shape[0]
        self.pdim = 1 if self.nclass <= 2 else self.nclass

        if not (isinstance(x_train, np.ndarray) or isinstance(x_train, spmatrix)):
            x_train = np.array(x_train)
        if not (isinstance(y_train, np.ndarray) or isinstance(y_train, spmatrix)):
            y_train = np.array(y_train)
        if not (isinstance(x_test, np.ndarray) or isinstance(x_test, spmatrix)) and x_test is not None:
            x_test = np.array(x_test)
        if not (isinstance(x_probe, np.ndarray) or isinstance(x_probe, spmatrix)) and x_probe is not None:
            x_probe = np.array(x_probe)
        if not (isinstance(y_probe, np.ndarray) or isinstance(y_probe, spmatrix)) and y_probe is not None:
            y_probe = np.array(y_probe)
        if len(y_train.shape) == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()

        if self.verbose:
            if x_test is not None and x_probe is None:
                print 'Bagger: X_train: {0}, X_test: {1}'.format(x_train.shape, x_test.shape)
            elif x_test is None and x_probe is not None:
                print 'Bagger: X_train: {0}, X_probe: {1}'.format(x_train.shape, x_probe.shape)
            else:
                print 'Bagger: X_train: {0}, X_test: {1}, X_probe: {2}'. \
                    format(x_train.shape, x_test.shape, x_probe.shape)

        ntrain = x_train.shape[0]
        ntest = 0 if x_test is None else x_test.shape[0]
        nprobe = 0 if x_probe is None else x_probe.shape[0]

        yhat_test = np.zeros((ntest, self.pdim))
        yhat_probe = np.zeros((nprobe, self.pdim))

        if self.verbose:
            print '{0} Bags (seed: {1}, subsample: {2}, bootstrap: {3}, shuffle: {4})' \
                .format(self.nbags, self.seed, self.subsample, self.bootstrap, self.shuffle)

        bag_test = np.empty((self.nbags, ntest, self.pdim))
        bag_probe = np.empty((self.nbags, nprobe, self.pdim))

        bag_scores_probe = []

        for k in range(self.nbags):
            ts_bag = datetime.now()
            ix = np.random.choice(ntrain, int(self.subsample * ntrain), self.bootstrap)
            if self.verbose:
                print 'Bag {0:02d}:  X_train: {1}'.format(k + 1, x_train[ix].shape)
            if not self.shuffle:
                ix = np.sort(ix)
            clf = self.clf(params=self.clf_params.copy(), seed=self.seed + k)
            weights = sample_weights[ix] if sample_weights is not None else None
            clf.train(x_train[ix], y_train[ix], sample_weights=weights)
            if ntest > 0:
                bag_test[k, :, :] = clf.predict(x_test).reshape((-1, self.pdim))
            if nprobe > 0:
                bag_probe[k, :, :] = clf.predict(x_probe).reshape((-1, self.pdim))
                scr_probe = self.metric(y_probe, bag_probe[k, :, :])
                bag_scores_probe.append(scr_probe)
            te_bag = datetime.now()
            if self.verbose and nprobe > 0:
                print u'         {0:.12f} ({1:.8f} ± {2:.8f})'. \
                    format(scr_probe, np.mean(bag_scores_probe), np.std(bag_scores_probe))
            if self.verbose:
                print '         {0}'.format((te_bag - ts_bag))
        if ntest > 0:
            yhat_test[:, :] = bag_test.mean(axis=0)
        if nprobe > 0:
            yhat_probe[:, :] = bag_probe.mean(axis=0)

        te = datetime.now()
        elapsed_time = (te - ts)

        self.yhat_test = yhat_test
        self.yhat_probe = yhat_probe
        self.elapsed_time = elapsed_time
        self.probe_score = self.metric(y_probe, yhat_probe) if self.nprobe > 0 else None

        if self.verbose:
            if self.nprobe > 0:
                print 'Holdout: {0:.12f}'.format(self.probe_score)
            print 'Runtime: {0}'.format(elapsed_time)

    @property
    def test_predictions(self):
        return self.yhat_test

    @property
    def probe_predictions(self):
        return self.yhat_probe
