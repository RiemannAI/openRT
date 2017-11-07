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
    
    def getinit(self, S, mean=0, sdev=1):
        
        return np.random.normal(mean, sdev, S)
    
class null(initializer):
    
    def getinit(self, S):
        
        return np.zeros(S)    
    
class glorot_normal(initializer):
    def getinit(self, S):
        return np.random.normal(0, 2.0/(S[0]+S[1]),S)
    
class glorot_uniform(initializer):
    def getinit(self, S):
        limit = np.sqrt(6.0)/(S[0]+S[1])
                        
        return np.random.uniform(-limit,limit,S)