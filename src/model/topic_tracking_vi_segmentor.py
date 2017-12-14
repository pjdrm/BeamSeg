'''
Created on Nov 1, 2017

@author: pjdrm
'''
import numpy as np
from dataset.synthetic_doc_cvb import CVBSynDoc

class TopicTrackingVIModel(object):

    def __init__(self, alpha, beta, doc):
        self.beta = beta
        self.beta_sum = np.sum(self.beta)
        self.alpha = alpha
        self.K = len(self.alpha)
        self.W = doc.W
        self.doc = doc
        self.n_words = np.sum(self.sents_len)
        #TODO: this probably does not work real docs because they are sentence based
        self.I_words = self.doc.I_words
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
            
    def get_word_topics(self):
        '''
        Returns the word topic assignments. The final topic
        for a given word is the variational parameter with the 
        highest value.
        '''
        word_topics = []
        for i in range(self.n_words):
            word_topics.append(np.argmax(self.gamma_q[i]))
        return word_topics
        
K = 10
W = 30
alpha = [15]*K
beta = [0.6]*W
n_words = 1000

doc_synth = CVBSynDoc(alpha, beta, n_words)
vi_tt_model = TopicTrackingVIModel(alpha, beta, doc_synth)   
        