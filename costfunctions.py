#!/usr/bin/env sage
# -*- coding: utf-8 -*-

import numpy as np


def logarithmic(x, *y):
    """ Logarithmic total cost """
    res = -np.sum(np.log(x))
    return res


def mse(x, y):
    """ Mean squared error """
    return np.mean((x-y)**2)

def rmse(x, y):
    """ Root mean squared error """
    return np.sqrt(np.mean((x-y)**2))
