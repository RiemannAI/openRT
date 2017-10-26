# -*- coding: utf-8 -*-

import numpy as np


def logarithmic(x, *y):
    """ Logarithmic total cost """
    return -np.sum(np.log(x))


def sum(x, *y):
    """ Sum total cost """
    return -np.sum(x)


def mse(x, y):
    """ Mean squared error """
    return np.mean((x-y)**2)


def rmse(x, y):
    """ Root mean squared error """
    return np.sqrt(np.mean((x-y)**2))


def crossentropy(x,y):
    """ cross-entropy """
    
    lx  = np.log(x)
    lmx = np.log(1-x)
   
    return -np.mean(np.multiply(y,lx)+np.multiply(1-y,lmx))
    