# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

import xgboost as xgb


def create_feature_map(filename, features):
    outfile = open(filename, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_predictions_from_dump(xgb_model_file, xgdata, ntree_limit):
    gbdt = xgb.Booster()
    gbdt.load_model(xgb_model_file)
    return gbdt.predict(xgdata, ntree_limit=ntree_limit)


def get_split_value_histogram(xgb_model, feature, fmap='', bins=None):
    xgdump = xgb_model.get_dump(fmap=fmap)
    values = []
    regexp = re.compile("\[{0}<([\d.Ee+-]+)\]".format(feature))
    for i in range(len(xgdump)):
        m = re.findall(regexp, xgdump[i])
        values.extend(map(float, m))
    hist = pd.DataFrame.from_dict(Counter(values), orient='index'). \
        reset_index(). \
        rename(columns={'index': 'SplitValue', 0: 'Count'}). \
        sort_values(by='SplitValue').reset_index(drop=True)

    if bins is not None:
        nph = np.histogram(hist.SplitValue, bins=bins, weights=hist.Count)
        hist = pd.DataFrame()
        hist['SplitValue'] = nph[1][1:]
        hist['Count'] = nph[0]

    hist.columns.name = feature
    return hist
