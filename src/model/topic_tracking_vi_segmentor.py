'''
Created on Nov 1, 2017

@author: pjdrm
'''
import numpy as np
from dataset.synthetic_doc import SyntheticTopicTrackingDoc
from debug.synthetic_corpus_debugger import print_corpus
from scipy.special import gamma, digamma
import numpy_groupies as npg

class TopicTrackingVIModel(object):

    def __init__(self, alpha, beta, K, doc):
        self.beta = beta
        self.beta_sum = np.sum(self.beta)
        self.alpha = alpha
        self.K = K
        self.W = doc.W
        self.doc = doc
        self.n_words = np.sum(self.sents_len)
        #TODO: this probably does not work real docs because sentence length varies
        self.I_words = self.doc.U_I_words.flatten()
        self.all_wi_dic = {} #keys are vocab indexes and value is the list of words of that type
        for i, word in enumerate(self.I_words):
            if word not in self.all_wi_dic:
                self.all_wi_dic[word] = []
            self.all_wi_dic.append(i)
        
        
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
        gamma_q = np.zeros((self.n_words, self.K))
        for i in range(self.n_words):
            gamma_q[i] = np.random.dirichlet(self.alpha)
        return gamma_q
    
    def cvb_iter(self):
        '''
        Performs a single iteration of the
        Collapsed Variational Bayes algorithm. That is,
        performs one round of updates for all variational 
        parameters.
        '''
        for i in range(self.n_words):
            # I think the updates are dependant on each other, we need to do one at a time
            E_q_zi_k = np.sum(self.gamma_q, axis=0)-self.gamma_q[i]
            Var_q_zi_k = E_q_zi_k*(1.0-E_q_zi_k)
            
            
            gamma_q_all_wi = self.gamma_q[self.all_wi_dic[self.I_words[i]]]
            q_wi_k = np.sum(gamma_q_all_wi, axis=0)
            E_q_wi_k = (q_wi_k-self.gamma_q)*self.W_I_counts_minus1
            Var_q_wi_k = E_q_wi_k*(1.0-E_q_wi_k)
            
            q_wi_k_plus_beta = q_wi_k + self.beta
            E_q_wi_k_plus_beta = (q_wi_k_plus_beta[self.I_words]-self.gamma_q)*self.W_I_counts_minus1
            
            f1 = E_q_zi_k + self.alpha
            f2 = E_q_wi_k_plus_beta
            f3 = 1.0/(E_q_zi_k+self.beta_sum)
            f4 = Var_q_zi_k/(2.0*f1)**2
            f5 = Var_q_wi_k/(2.0*f2)**2
            f6 = f1/(2.0*(E_q_zi_k+self.beta_sum))**2
            
            self.gamma_q[i] = f1*f2*f3*np.e(-f4-f5+f6)
        
        
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
        