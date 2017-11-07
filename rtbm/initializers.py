# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
import sys

class initializer(object):
    """ Abstract class for cost functions """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def getinit(self, S, *bound):
        pass
    
    
class uniform(initializer):
    
    def getinit(self, S, bound):
        return np.random.uniform(-bound, bound, S)
    
class normal(initializer):
    
    def getinit(self, S, mean=0,sdev=1):
        
        return np.random.normal(mean, sdev, S)