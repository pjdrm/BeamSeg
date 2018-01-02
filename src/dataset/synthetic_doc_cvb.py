'''
Created on Dec 14, 2017

@author: Pedro Mota
'''
import numpy as np
from scipy import int32
import copy

class CVBSynDoc(object):
    '''
    classdocs
    '''
    def __init__(self, beta, pi, sent_len, doc_len, n_docs):
        self.isMD = False if n_docs == 1 else True
        self.n_docs = n_docs
        n_sents = doc_len*n_docs
        self.W = len(beta)
        self.rho = np.random.binomial(1, pi, size=n_sents)
        #The last sentence of each document must be 1
        for u_end in range(doc_len-1, n_sents, doc_len):
            self.rho[u_end] = 1
        #... except last sentence.
        self.rho[-1] = 0
        self.docs_index = range(doc_len, n_sents+1, doc_len)
        self.K = np.count_nonzero(self.rho)+1
        self.phi = np.array([np.random.dirichlet(beta) for k in range(self.K)])
            
        self.U_W_counts = np.zeros((n_sents, self.W), dtype=int32)
        k = 0
        for u_end in range(len(self.rho)):
            word_counts = np.random.multinomial(sent_len, self.phi[k], size=1)
            self.U_W_counts[u_end] = word_counts
            if self.rho[u_end] == 1:
                k += 1
            if u_end in self.docs_index:
                k = 0
                
    def get_single_docs(self):
        doc_l = []
        doc_begin = 0
        for doc_end in self.docs_index:
            doc = copy.deepcopy(self)
            doc.n_sents = doc_end - doc_begin
            doc.n_docs = 1
            doc.docs_index = [doc.n_sents]
            doc.rho = doc.rho[doc_begin:doc_end]
            doc.rho[-1] = 0
            doc.U_W_counts = doc.U_W_counts[doc_begin:doc_end, :]
            doc.isMD = False
            doc_begin = doc_end
            doc_l.append(doc)
        return doc_l