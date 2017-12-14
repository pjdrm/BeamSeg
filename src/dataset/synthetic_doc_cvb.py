'''
Created on Dec 14, 2017

@author: Pedro Mota
'''
import numpy as np

class CVBSynDoc(object):
    '''
    classdocs
    '''
    def __init__(self, alpha, beta, n_words):
        self.W = len(beta)
        self.K = len(alpha)
        self.theta = np.random.dirichlet(alpha)
        self.phi = []
        for k in range(self.K):
            self.phi.append(np.random.dirichlet(beta))
            
        self.I_words = []
        for wi in range(n_words):
            k = np.nonzero(np.random.multinomial(1, self.theta, size=1))[1][0]
            word = np.nonzero(np.random.multinomial(1, self.phi[k], size=1))[1][0]
            self.I_words.append(word)