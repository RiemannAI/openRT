# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
import sys

class actfunc(object):
    """ Abstract class for cost functions """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def activation(self, X):
        pass
    
    @abstractmethod
    def gradient(self, X):
        pass
    
class linear(actfunc):
    """ linear pass through """
    
    def activation(self, x):
        return x

    def gradient(self, x):
        return np.ones(x.shape)
    
class sigmoid(actfunc):
    """ The sigmoid """

    def activation(self, x):   
        return 1.0/(1+np.exp(-x))
   
    def gradient(self, x):
        return cost(x)*(1-cost(x))
    
        
