'''
Created on Feb 9, 2017

@author: root
'''
from functools import lru_cache
from scipy.special import gammaln

@lru_cache(maxsize=None)
def gammaln_cache(x):
    return gammaln(x)
