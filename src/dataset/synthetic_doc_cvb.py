'''
Created on Dec 14, 2017

@author: Pedro Mota
'''
import numpy as np
from scipy import int32

class CVBSynDoc(object):
    '''
    classdocs
    '''
    def __init__(self, beta, pi, sent_len, doc_len, n_docs):
        self.n_docs = n_docs
        n_sents = doc_len*n_docs
        self.W = len(beta)
        self.rho = np.random.binomial(1, pi, size=n_sents)
        #The last sentence of each document must be 1
        for u in range(doc_len-1, n_sents, doc_len):
            self.rho[u] = 1
        #... except last sentence.
        self.rho[-1] = 0
        self.docs_index = range(doc_len, n_sents+1, doc_len)
        self.K = np.count_nonzero(self.rho)+1
        self.phi = np.array([np.random.dirichlet(beta) for k in range(self.K)])
            
        self.U_W_counts = np.zeros((n_sents, self.W), dtype=int32)
        k = 0
        for u in range(len(self.rho)):
            word_counts = np.random.multinomial(sent_len, self.phi[k], size=1)
            self.U_W_counts[u] = word_counts
            if self.rho[u] == 1:
                k += 1
            if u in self.docs_index:
                k = 0
