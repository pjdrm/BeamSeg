'''
Created on Nov 1, 2017

@author: pjdrm
'''
import numpy as np
from dataset.synthetic_doc import SyntheticTopicTrackingDoc
from debug.synthetic_corpus_debugger import print_corpus
from scipy.special import gamma, digamma

class TopicTrackingVIModel(object):

    def __init__(self, alpha, beta, K, doc):
        self.beta = beta
        self.alpha = alpha
        self.K = K
        self.W = doc.W
        self.doc = doc

        '''
        Array with the length of each sentence.
        Note that the length of the sentences is variable
        '''
        self.sents_len = self.doc.sents_len
        self.n_sents = self.doc.n_sents
        
        '''
        Variational Parameters
        Each word i has a topic assignment z_i variational
        parameters gamma_i. Each gamma_i is K-dim vector (each
        index is a topic).         
        '''
        self.gamma_q = self.init_gamma_q()

    def init_gamma_q(self):
        '''
        Initializes the gamma variational parameters of z.
        Initialization cannot be uniform, we draw multinomials
        from a Dirichlet to initialize each gamma_i.
        '''
        n_words = np.sum(self.sents_len)
        gamma_q = np.zeros((n_words, self.K))
        for i in range(n_words):
            gamma_q[i] = np.random.dirichlet(self.alpha)
        return gamma_q
        
pi = 0.2
alpha = 15
beta = 0.6
K = 10
W = 15
n_sents = 50
sentence_l = 50
vocab_dic = {}
for w in range(W):
    vocab_dic[w] = "w" + str(w)
outDir = "debug/synthetic_dataset/"

print_theta_flag = False
print_heat_map_flag = True
print_text_flag = False
flags = [print_theta_flag, print_heat_map_flag, print_text_flag]

doc_synth_tt = SyntheticTopicTrackingDoc(pi, alpha, beta, K, W, n_sents, sentence_l)
doc_synth_tt.generate_docs(10)
print_corpus(vocab_dic, doc_synth_tt, "Topic Tracking", outDir, flags)

vi_tt_model = TopicTrackingVIModel([alpha]*K, [beta]*K, K, doc_synth_tt)      
        