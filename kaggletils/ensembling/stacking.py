# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""

import abc
import copy_reg
import types
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.sparse import spmatrix, csr_matrix
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import log_loss


def __pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copy_reg.pickle(types.MethodType, __pickle_method)


class CrossValidatorClfBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, params=None, seed=0):
        return

    @abc.abstractmethod
    def train(self, x_train, y_train, x_valid=None, y_valid=None, sample_weights=None):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass


class CrossValidator(object):
    def __init__(self, clf, clf_params=None, nfolds=5, folds=None, stratified=True, seed=0, regression=False, nbags=1,
                 subsample=1., bootstrap=False, shuffle=False, metric=log_loss, average_oof=False, verbose=True):
        self.clf = clf
        self.clf_params = clf_params if clf_params is not None else {}
        self.nfolds = nfolds if folds is None else len(folds)
        self.stratified = None if folds is not None or nfolds < 2 else False if regression is True else stratified
        self.seed = seed
        self.regression = regression
        self.nbags = nbags
        self.subsample = subsample
        self.bootstrap = bootstrap
        self.shuffle = shuffle
        self.metric = metric
        self.verbose = verbose
        self.average_oof = average_oof if nfolds > 1 else False
        self.nclass = None
        self.pdim = None
        self.sample_weights = None
        self.subsets = False

        self.oof_train = None
        self.oof_test = None
        self.oof_probe = None
        self.elapsed_time = None
        self.cv_scores = None
        self.cv_mean = None
        self.cv_std = None
        self.cv_scores_probe = None
        self.cv_mean_probe = None
        self.cv_std_probe = None
        self.folds = folds
        self.mean_train = None
        self.mean_test = None
        self.mean_probe = None

        self.train_score = None
        self.probe_score = None

        self.ntrain = None
        self.ntest = None
        self.nprobe = None

    def run_cv(self, x_train, y_train, x_test=None, x_probe=None, y_probe=None,
               sample_weights=None, subset_column=None, subset_values=None, ccv_features=None):
        ts = datetime.now()
        self.ntrain = x_train.shape[0]
        self.ntest = 0 if x_test is None else x_test.shape[0]
        self.nprobe = 0 if x_probe is None else x_probe.shape[0]
        self.sample_weights = sample_weights
        self.nclass = 1 if self.regression else np.unique(y_train).shape[0]
        self.pdim = 1 if self.nclass <= 2 else self.nclass
        self.sparse_input = isinstance(x_train, spmatrix)

        if not (isinstance(x_train, np.ndarray) or isinstance(x_train, spmatrix)):
            subset_column = x_train.columns.get_loc(subset_column) if subset_column is not None else None
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
            if x_test is None and x_probe is None:
                print 'CrossValidator: X_train: {0}'.format(x_train.shape)
            elif x_test is not None and x_probe is None:
                print 'CrossValidator: X_train: {0}, X_test: {1}'.format(x_train.shape, x_test.shape)
            elif x_test is None and x_probe is not None:
                print 'CrossValidator: X_train: {0}, X_probe: {1}'.format(x_train.shape, x_probe.shape)
            else:
                print 'CrossValidator: X_train: {0}, X_test: {1}, X_probe: {2}'. \
                    format(x_train.shape, x_test.shape, x_probe.shape)

        if subset_column is None:
            self.__run_cv(x_train, y_train, x_test, x_probe, y_probe, sample_weights, ccv_features)
            return

        self.subsets = True
        oof_train = np.zeros((self.ntrain, self.pdim))
        oof_test = np.zeros((self.ntest, self.pdim))
        oof_probe = np.zeros((self.nprobe, self.pdim))

        if subset_values is None:
            subset_values = np.unique(x_train[:, subset_column])

        cv_scores = []
        cv_scores_weights = []
        cv_scores_probe = []
        cv_scores_probe_weights = []

        if self.verbose:
            print '{0} Subsets (subset column: {1}, subset values: {2})' \
                .format(len(subset_values), subset_column, subset_values)

        for val in subset_values:
            train_ix = np.in1d(x_train[:, subset_column], val)
            test_ix = np.in1d(x_test[:, subset_column], val) if self.ntest > 0 else None
            probe_ix = np.in1d(x_probe[:, subset_column], val) if self.nprobe > 0 else None
            x_train_sub = x_train[train_ix]
            y_train_sub = y_train[train_ix]
            x_test_sub = x_test[test_ix] if self.ntest > 0 else None
            x_probe_sub = x_probe[probe_ix] if self.nprobe > 0 else None
            y_probe_sub = y_probe[probe_ix] if self.nprobe > 0 else None
            weights_sub = sample_weights[train_ix] if sample_weights is not None else None
            if self.verbose:
                if self.ntest > 0:
                    print 'Subset CV (column: {0}, value: {1}, ntrain: {2}, ntest: {3})' \
                        .format(subset_column, val, x_train_sub.shape[0], x_test_sub.shape[0])
                else:
                    print 'Subset CV (column: {0}, value: {1}, ntrain: {2})' \
                        .format(subset_column, val, x_train_sub.shape[0])
            self.folds = None
            self.__run_cv(x_train_sub, y_train_sub, x_test_sub, x_probe_sub, y_probe_sub, weights_sub)
            oof_train[train_ix] = self.oof_train
            oof_test[test_ix] = self.oof_test
            oof_probe[probe_ix] = self.oof_probe
            cv_scores.append(self.cv_scores)
            cv_scores_weights.append(x_train_sub.shape[0])
            cv_scores_probe.append(self.cv_scores_probe)
            if self.nprobe > 0:
                cv_scores_probe_weights.append(x_probe_sub.shape[0])

        cv_scores = np.average(cv_scores, axis=0, weights=cv_scores_weights)
        if self.nprobe > 0:
            cv_scores_probe = np.average(cv_scores_probe, axis=0, weights=cv_scores_probe_weights)

        te = datetime.now()
        elapsed_time = (te - ts)

        self.oof_train = oof_train
        self.oof_test = oof_test
        self.oof_probe = oof_probe
        self.mean_train = np.mean(oof_train, axis=0)
        self.mean_test = np.mean(oof_test, axis=0) if self.ntest > 0 else None
        self.mean_probe = np.mean(oof_probe, axis=0) if self.nprobe > 0 else None
        self.cv_scores = cv_scores
        self.cv_mean = np.mean(cv_scores)
        self.cv_std = np.std(cv_scores)
        self.cv_scores_probe = cv_scores_probe
        self.cv_mean_probe = np.mean(cv_scores_probe)
        self.cv_std_probe = np.std(cv_scores_probe)
        self.elapsed_time = elapsed_time
        self.train_score = self.metric(y_train, oof_train)
        self.probe_score = self.metric(y_probe, oof_probe) if self.nprobe > 0 else None

        if self.nclass <= 2:
            self.mean_train = self.mean_train[0]
            self.mean_test = self.mean_test[0] if self.ntest > 0 else None

        if self.verbose:
            print 'CV-Mean: {0:.12f}'.format(self.cv_mean)
            print 'CV-Std:  {0:.12f}'.format(self.cv_std)
            print 'Runtime: {0}'.format(elapsed_time)

    def __run_cv(self, x_train, y_train, x_test=None, x_probe=None, y_probe=None, sample_weights=None,
                 ccv_features=None):
        ts = datetime.now()
        ntrain = x_train.shape[0]
        ntest = 0 if x_test is None else x_test.shape[0]
        nprobe = 0 if x_probe is None else x_probe.shape[0]
        prefix = '\t' if self.subsets else ''

        if ccv_features is not None:
            if ntest > 0:
                x_test = np.column_stack((x_test, ccv_features[1][-1])) if not self.sparse_input \
                    else sparse.hstack((x_test, ccv_features[1][-1].reshape(x_test.shape[0], -1)), format='csr')
            if nprobe > 0:
                x_probe = np.column_stack((x_probe, ccv_features[2][-1])) if not self.sparse_input \
                    else sparse.hstack((x_probe, ccv_features[2][-1].reshape(x_probe.shape[0], -1)), format='csr')

        if self.folds is None:
            if self.nfolds > 1:
                if self.stratified:
                    self.folds = StratifiedKFold(y_train, n_folds=self.nfolds, shuffle=True, random_state=self.seed)
                else:
                    self.folds = KFold(ntrain, n_folds=self.nfolds, shuffle=True, random_state=self.seed)
            else:
                self.folds = [(np.arange(ntrain), [])]

        oof_train = np.zeros((ntrain, self.pdim))
        oof_test = np.zeros((ntest, self.pdim))
        oof_probe = np.zeros((nprobe, self.pdim))
        oof_test_folds = np.empty((self.nfolds, ntest, self.pdim))
        oof_probe_folds = np.empty((self.nfolds, nprobe, self.pdim))

        cv_scores = []
        cv_scores_probe = []

        if self.verbose:
            print prefix + '{0} Fold CV (seed: {1}, stratified: {2}, nbags: {3}, ' \
                           'subsample: {4}, bootstrap: {5}, shuffle: {6}, average oof: {7})' \
                .format(self.nfolds, self.seed, self.stratified, self.nbags,
                        self.subsample, self.bootstrap, self.shuffle, self.average_oof)

        if self.nfolds > 1:
            ts_cv = datetime.now()
            for i, (train_ix, valid_ix) in enumerate(self.folds):
                ts_fold = datetime.now()
                x_train_oof = x_train[train_ix]
                y_train_oof = y_train[train_ix]
                x_valid_oof = x_train[valid_ix]
                y_valid_oof = y_train[valid_ix]

                if ccv_features is not None:
                    x_train_oof = np.column_stack((x_train_oof, ccv_features[0][i])) if not self.sparse_input \
                        else sparse.hstack((x_train_oof, ccv_features[0][i].reshape(x_train_oof.shape[0], -1)), format='csr')
                    x_valid_oof = np.column_stack((x_valid_oof, ccv_features[1][i])) if not self.sparse_input \
                        else sparse.hstack((x_valid_oof, ccv_features[1][i].reshape(x_valid_oof.shape[0], -1)), format='csr')

                if self.verbose:
                    print prefix + 'Fold {0:02d}: X_train: {1}, X_valid: {2}'. \
                        format(i + 1, x_train_oof.shape, x_valid_oof.shape)

                ntrain_oof = x_train_oof.shape[0]
                nvalid_oof = x_valid_oof.shape[0]

                oof_bag_valid = np.empty((self.nbags, nvalid_oof, self.pdim))
                oof_bag_test = np.empty((self.nbags, ntest, self.pdim))
                oof_bag_probe = np.empty((self.nbags, nprobe, self.pdim))

                for k in range(self.nbags):
                    ix = np.random.choice(ntrain_oof, int(self.subsample * ntrain_oof), self.bootstrap)
                    if not self.shuffle:
                        ix = np.sort(ix)
                    weights = sample_weights[train_ix][ix] if sample_weights is not None else None
                    clf = self.clf(params=self.clf_params.copy(), seed=self.seed + k)
                    clf.train(x_train_oof[ix], y_train_oof[ix], x_valid_oof, y_valid_oof, weights)

                    oof_bag_valid[k, :, :] = clf.predict(x_valid_oof).reshape((-1, self.pdim))
                    if ntest > 0:
                        oof_bag_test[k, :, :] = clf.predict(x_test).reshape((-1, self.pdim))
                    if nprobe > 0:
                        oof_bag_probe[k, :, :] = clf.predict(x_probe).reshape((-1, self.pdim))

                pred_oof_valid = oof_bag_valid.mean(axis=0)
                oof_train[valid_ix, :] = pred_oof_valid
                if ntest > 0:
                    pred_oof_test = oof_bag_test.mean(axis=0)
                    oof_test_folds[i, :, :] = pred_oof_test
                if nprobe > 0:
                    pred_oof_probe = oof_bag_probe.mean(axis=0)
                    oof_probe_folds[i, :, :] = pred_oof_probe

                scr = self.metric(y_valid_oof, pred_oof_valid) if nvalid_oof > 0 else np.nan
                scr_probe = self.metric(y_probe, pred_oof_probe) if nprobe > 0 else np.nan

                cv_scores.append(scr)
                cv_scores_probe.append(scr_probe)

                te_fold = datetime.now()
                if self.verbose:
                    print prefix + u'         {0:.12f} ({1:.8f} ± {2:.8f})'. \
                        format(scr, np.mean(cv_scores), np.std(cv_scores))
                if self.verbose and nprobe > 0:
                    print prefix + u'         {0:.12f} ({1:.8f} ± {2:.8f})'. \
                        format(scr_probe, np.mean(cv_scores_probe), np.std(cv_scores_probe))
                if self.verbose:
                    print prefix + '         {0}'.format(te_fold - ts_fold)
            te_cv = datetime.now()
        else:
            ts_cv = te_cv = datetime.now()
            cv_scores = np.nan
            cv_scores_probe = np.nan

        self.cv_scores = cv_scores
        self.cv_mean = np.mean(cv_scores)
        self.cv_std = np.std(cv_scores)

        self.cv_scores_probe = cv_scores_probe
        self.cv_mean_probe = np.mean(cv_scores_probe)
        self.cv_std_probe = np.std(cv_scores_probe)

        if ntest > 0 or nprobe > 0:
            if self.average_oof:
                if ntest > 0:
                    oof_test[:, :] = oof_test_folds.mean(axis=0)
                if nprobe > 0:
                    oof_probe[:, :] = oof_probe_folds.mean(axis=0)
            else:
                if ccv_features is not None:
                    x_train = np.column_stack((x_train, ccv_features[0][-1])) if not self.sparse_input \
                        else sparse.hstack((x_train, ccv_features[0][-1].reshape(x_train.shape[0], -1)), format='csr')

                if self.verbose and self.nfolds > 1:
                    print prefix + 'CV-Mean: {0:.12f}'.format(self.cv_mean)
                    print prefix + 'CV-Std:  {0:.12f}'.format(self.cv_std)
                    print prefix + 'Runtime: {0}'.format((te_cv - ts_cv))
                if self.verbose:
                    print prefix + 'OnePass: X_train: {0}, X_test: {1}'. \
                        format(x_train.shape, x_test.shape)

                oof_bag_test = np.empty((self.nbags, ntest, self.pdim))
                oof_bag_probe = np.empty((self.nbags, nprobe, self.pdim))
                for k in range(self.nbags):
                    ix = np.random.choice(ntrain, int(self.subsample * ntrain), self.bootstrap)
                    if not self.shuffle:
                        ix = np.sort(ix)
                    clf = self.clf(params=self.clf_params.copy(), seed=self.seed + k)
                    weights = sample_weights[ix] if sample_weights is not None else None
                    clf.train(x_train[ix], y_train[ix], sample_weights=weights)
                    if ntest > 0:
                        oof_bag_test[k, :, :] = clf.predict(x_test).reshape((-1, self.pdim))
                    if nprobe > 0:
                        oof_bag_probe[k, :, :] = clf.predict(x_probe).reshape((-1, self.pdim))
                if ntest > 0:
                    oof_test[:, :] = oof_bag_test.mean(axis=0)
                if nprobe > 0:
                    oof_probe[:, :] = oof_bag_probe.mean(axis=0)

        te = datetime.now()
        elapsed_time = (te - ts)

        self.oof_train = oof_train
        self.oof_test = oof_test
        self.oof_probe = oof_probe
        self.mean_train = np.mean(oof_train, axis=0)
        self.mean_test = np.mean(oof_test, axis=0) if ntest > 0 else None
        self.mean_probe = np.mean(oof_probe, axis=0) if nprobe > 0 else None
        self.elapsed_time = elapsed_time
        self.train_score = self.metric(y_train, oof_train)
        self.probe_score = self.metric(y_probe, oof_probe) if self.nprobe > 0 else None

        if self.nclass <= 2:
            self.mean_train = self.mean_train[0]
            self.mean_test = self.mean_test[0] if ntest > 0 else None

        if self.verbose:
            if self.average_oof:
                print prefix + 'CV-Mean: {0:.12f}'.format(self.cv_mean)
                print prefix + 'CV-Std:  {0:.12f}'.format(self.cv_std)
            if self.nprobe > 0:
                print prefix + 'Holdout: {0:.12f}'.format(self.probe_score)
            print prefix + 'Runtime: {0}'.format(elapsed_time)

            # if self.verbose:
            #     for k in range(self.nfolds):
            #         print prefix + 'Fold {0:02d}: {1:.12f}'.format(k + 1, self.cv_scores[k])
            #     print prefix + 'CV-Mean: {0:.12f}'.format(self.cv_mean)
            #     print prefix + 'CV-Std:  {0:.12f}'.format(self.cv_std)
            #     print prefix + 'Runtime: {0}'.format(elapsed_time)

    def print_cv_summary(self):
        if self.cv_scores is None:
            return
        for k in range(self.nfolds):
            print 'Fold {0:02d}: {1:.12f}'.format(k + 1, self.cv_scores[k])
        print 'CV-Mean: {0:.12f}'.format(self.cv_mean)
        print 'CV-Std:  {0:.12f}'.format(self.cv_std)
        if self.nprobe > 0:
            print 'Holdout: {0:.12f}'.format(self.probe_score)
        print 'Runtime: {0}'.format(self.elapsed_time)

    @property
    def oof_predictions(self):
        return self.oof_train, self.oof_test

    @property
    def train_predictions(self):
        return self.oof_train

    @property
    def test_predictions(self):
        return self.oof_test

    @property
    def probe_predictions(self):
        return self.oof_probe

    @property
    def cv_stats(self):
        return self.cv_mean, self.cv_std

    @property
    def oof_means(self):
        return self.mean_train, self.mean_test

    @property
    def oof_means_delta(self):
        return np.abs(self.mean_train - self.mean_test)


class CrossValidatorMT(object):
    def __init__(self, clf, clf_params=None, nfolds=5, stratified=True, seed=0, regression=False, nbags=1,
                 metric=log_loss, average_oof=False, verbose=True):
        self.clf = clf
        self.clf_params = clf_params
        self.nfolds = nfolds
        self.stratified = stratified
        self.seed = seed
        self.regression = regression
        self.nbags = nbags
        self.metric = metric
        self.verbose = verbose
        self.average_oof = average_oof
        self.nclass = None
        self.pdim = None
        self.sample_weights = None

        self.oof_train = None
        self.oof_test = None
        self.elapsed_time = None
        self.cv_scores = None
        self.cv_mean = None
        self.cv_std = None
        self.folds = None
        self.mean_train = None
        self.mean_test = None

        self.x_train = None
        self.y_train = None
        self.x_test = None

        self.ntrain = None
        self.ntest = None

    def run_cv(self, x_train, y_train, x_test=None, sample_weights=None):
        ts = datetime.now()
        if not isinstance(x_train, np.ndarray):
            x_train = np.array(x_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
        if not isinstance(x_test, np.ndarray) and x_test is not None:
            x_test = np.array(x_test)

        self.sample_weights = sample_weights
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

        if self.verbose:
            if x_test is None:
                print 'CrossValidatorMT {0}'.format(x_train.shape)
            else:
                print 'CrossValidatorMT {0} {1}'.format(x_train.shape, x_test.shape)

        pool = Pool(processes=self.nfolds)

        self.ntrain = x_train.shape[0]
        self.ntest = 0 if x_test is None else x_test.shape[0]

        self.nclass = 1 if self.regression else np.unique(y_train).shape[0]
        self.pdim = 1 if self.nclass <= 2 else self.nclass

        if self.stratified:
            folds = StratifiedKFold(y_train, n_folds=self.nfolds, shuffle=True, random_state=self.seed)
        else:
            folds = KFold(self.ntrain, n_folds=self.nfolds, shuffle=True, random_state=self.seed)

        oof_train = np.zeros((self.ntrain, self.pdim))
        oof_test = np.zeros((self.ntest, self.pdim))
        oof_test_folds = np.empty((self.nfolds, self.ntest, self.pdim))

        cv_scores = []

        if self.verbose:
            print '{0} Fold CV (seed: {1}, stratified: {2}, nbags: {3}, average oof: {4})' \
                .format(self.nfolds, self.seed, self.stratified, self.nbags, self.average_oof)

        ts_fold = datetime.now()
        folds_oof = pool.map(self._process_fold, folds)
        te_fold = datetime.now()

        for i, (train_index, valid_index) in enumerate(folds):
            y_valid_oof = y_train[valid_index]

            scr = self.metric(y_valid_oof, folds_oof[i][0])
            if self.verbose:
                print 'Fold {0:02d}: {1:.12f} ({2})'.format(i + 1, scr, (te_fold - ts_fold))

            cv_scores.append(scr)
            oof_train[valid_index, :] = folds_oof[i][0]
            oof_test_folds[i, :, :] = folds_oof[i][1]

        if self.ntest > 0:
            if self.average_oof:
                oof_test[:, :] = oof_test_folds.mean(axis=0)
            else:
                oof_bag_test = np.empty((self.nbags, self.ntest, self.pdim))
                for k in range(self.nbags):
                    clf = self.clf(params=self.clf_params.copy(), seed=self.seed + k)
                    clf.train(x_train, y_train, sample_weights=self.sample_weights)
                    oof_bag_test[k, :, :] = clf.predict(x_test).reshape((-1, self.pdim))
                oof_test[:, :] = oof_bag_test.mean(axis=0)

        te = datetime.now()
        elapsed_time = (te - ts)

        self.oof_train = oof_train
        self.oof_test = oof_test
        self.cv_scores = cv_scores
        self.cv_mean = np.mean(cv_scores)
        self.cv_std = np.std(cv_scores)
        self.elapsed_time = elapsed_time
        self.folds = folds
        self.mean_train = np.mean(oof_train, axis=0)
        self.mean_test = np.mean(oof_test, axis=0) if self.ntest > 0 else None

        if self.verbose:
            print 'CV-Mean: {0:.12f}'.format(self.cv_mean)
            print 'CV-Std:  {0:.12f}'.format(self.cv_std)
            print 'Runtime: {0}'.format(elapsed_time)

    def _process_fold(self, fold):
        train_ix, valid_ix = fold
        x_train_oof = self.x_train[train_ix]
        y_train_oof = self.y_train[train_ix]
        x_valid_oof = self.x_train[valid_ix]
        y_valid_oof = self.y_train[valid_ix]
        weights = self.sample_weights[train_ix] if self.sample_weights is not None else None

        nvalid_oof = x_valid_oof.shape[0]

        oof_bag_valid = np.empty((self.nbags, nvalid_oof, self.pdim))
        oof_bag_test = np.empty((self.nbags, self.ntest, self.pdim))

        for k in range(self.nbags):
            clf = self.clf(params=self.clf_params.copy(), seed=self.seed + k)
            clf.train(x_train_oof, y_train_oof, x_valid_oof, y_valid_oof, sample_weights=weights)

            oof_bag_valid[k, :, :] = clf.predict(x_valid_oof).reshape((-1, self.pdim))
            if self.ntest > 0:
                oof_bag_test[k, :, :] = clf.predict(self.x_test).reshape((-1, self.pdim))

        pred_oof_valid = oof_bag_valid.mean(axis=0)
        pred_oof_test = oof_bag_test.mean(axis=0) if self.ntest > 0 else None

        return pred_oof_valid, pred_oof_test

    @property
    def oof_predictions(self):
        return self.oof_train, self.oof_test

    @property
    def cv_stats(self):
        return self.cv_mean, self.cv_std

    @property
    def oof_means(self):
        return self.mean_train, self.mean_test
