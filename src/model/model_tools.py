'''
Created on Feb 9, 2017

@author: root
'''
from scipy.special import gammaln
import time

gammaln_cache_dic = {}
def gammaln_cache(x):
    if x not in gammaln_cache_dic:
        val = gammaln(x)
        gammaln_cache_dic[x] = val
    else:
        val = gammaln_cache_dic[x]
    return val

def cache_gammaln_mat_sum(M):
    row = M.shape[0]
    col = M.shape[1]
    res_sum = 0.0
    for i,j in zip(range(row), range(col)):
        x = M[i,j]
        res_sum += gammaln_cache(x)
    return res_sum

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
