'''
Created on Feb 9, 2017

@author: root
'''
from functools import lru_cache
from scipy.special import gammaln
import time

@lru_cache(maxsize=None)
def gammaln_cache(x):
    return gammaln(x)

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % (self.msecs))
