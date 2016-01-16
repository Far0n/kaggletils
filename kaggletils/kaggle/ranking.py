# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""

import numpy as np


def kaggle_points(rank=1, nteams=1, team_size=1, t=0.0):
    """
    :param rank: ladder rank
    :param nteams: number of teams in competition
    :param team_size: team size
    :param t: time since points gained

    :returns kaggle ranking points for ranking system since 13 May 2015

    :type rank: int
    :type nteams: int
    :type team_size: int
    :type t float

    :rtype: float
    """
    return (100000. / np.sqrt(team_size)) * (rank ** -0.75) * (np.log10(1 + np.log10(nteams))) * np.exp(-t / 500.)
