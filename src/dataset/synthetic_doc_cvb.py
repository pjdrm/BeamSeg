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
            if u+1 in self.docs_index:
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
    
class CVBSynDoc2(object):
    '''
    This version forces all documents to have the same number of segments
    '''
    def __init__(self, beta, pi, sent_len, n_segs, n_docs):
        self.isMD = False if n_docs == 1 else True
        self.n_docs = n_docs
        self.W = len(beta)
        self.rho = []
        self.docs_index = []
        n_sents = 0
        for doc_i in range(n_docs):
            for seg in range(n_segs):
                while not np.random.binomial(1, pi):
                    n_sents += 1
                    self.rho.append(0)
                n_sents += 1
                self.rho.append(1)
            self.docs_index.append(n_sents)
        self.rho[-1] = 0
        self.rho = np.array(self.rho)
        self.K = n_segs
        self.phi = np.array([np.random.dirichlet(beta) for k in range(self.K)])
        
        self.W_I_words = []
        doc_i = 0
        self.d_u_wi_indexes = [[] for i in range(self.n_docs)]
        self.U_W_counts = np.zeros((n_sents, self.W), dtype=int32)
        k = 0
        wi = 0
        for u in range(len(self.rho)):
            wi_list = []
            for word in range(sent_len):
                word_counts = np.random.multinomial(1, self.phi[k])
                w = np.nonzero(word_counts)[0][0] #w is a vocabulary index
                wi_list.append(wi)
                wi += 1
                self.W_I_words.append(w)
                self.U_W_counts[u] += word_counts
            
            self.d_u_wi_indexes[doc_i].append(wi_list)    
            if self.rho[u] == 1:
                k += 1
                
            if u+1 in self.docs_index:
                k = 0
                doc_i += 1
                
        self.W_I_words = np.array(self.W_I_words)
                
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
    
class CVBSynDoc3(object):
    '''
    This version forces all documents to have the same number of segments
    '''
    def __init__(self, beta):
        self.isMD = True
        self.n_docs = 2
        self.W = len(beta)
        self.rho = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.rho += [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.docs_index = [9, 18]
        self.rho = np.array(self.rho)
        self.K = 2
        self.phi = np.array([np.random.dirichlet(beta) for k in range(self.K)])
        n_sents = len(self.rho)
        self.U_W_counts = np.zeros((n_sents, self.W), dtype=int32)
        sent_len = 10
        for u in range(len(self.rho)):
            word_counts = np.random.multinomial(sent_len, self.phi[0], size=1)
            self.U_W_counts[u] = word_counts
        self.U_W_counts[3] = np.random.multinomial(sent_len, self.phi[1], size=1)
                
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