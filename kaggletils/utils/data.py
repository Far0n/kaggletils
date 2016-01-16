# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""

import numpy as np
import pandas as pd
import ntpath
import os


def is_numpy(x):
    return isinstance(x, np.ndarray)


def is_pandas(x):
    return isinstance(x, pd.DataFrame)


def csv_to_pyarray(csv_in, file_out=None, array_name=None, enquote_elements=True, header=None, wrap=50):
    csv_filename, csv_file_extension = os.path.splitext(csv_in)
    csv_basename = ntpath.basename(csv_in).replace(csv_file_extension, '')
    file_out = file_out or csv_in.replace(csv_file_extension, '.py')
    array_name = array_name or csv_basename

    data = np.array(pd.read_csv(csv_in, header=header)).ravel()
    pyarray = '{0} = ['.format(array_name)
    length = 0
    for i, x in enumerate(data):
        length += len(str(x))
        if length > wrap:
            pyarray = "{0}{1}".format(pyarray, '\n')
            length = 0
        pyarray = "{0}'{1}', ".format(pyarray, x) if enquote_elements else "{0}{1}, ".format(pyarray, x)

    pyarray = '{0}]'.format(pyarray.rstrip()[:-1])
    with open(file_out, "w") as text_file:
        text_file.write("{0}".format(pyarray))


class TrainTestHelper(object):
    def __init__(self):
        self.ntrain = None

    def combine(self, train, test):
        self.ntrain = train.shape[0]
        if is_numpy(train):
            return np.row_stack((train, test))
        else:
            return pd.concat((train, test), axis=0).reset_index(drop=True)

    def split(self, train_test):
        if self.ntrain is None:
            return None
        if is_numpy(train_test):
            train = train_test[:self.ntrain, :]
            test = train_test[self.ntrain:, :]
        else:
            train = train_test.iloc[:self.ntrain, :].copy().reset_index(drop=True)
            test = train_test.iloc[self.ntrain:, :].copy().reset_index(drop=True)
        return train, test
