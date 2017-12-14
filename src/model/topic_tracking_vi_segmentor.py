'''
Created on Nov 1, 2017

@author: pjdrm
'''
import numpy as np
from tqdm import trange
from dataset.synthetic_doc_cvb import CVBSynDoc
from sklearn.metrics.cluster import adjusted_rand_score

class TopicTrackingVIModel(object):

    def __init__(self, alpha, beta, doc, n_cvb_iters=20):
        self.beta = beta
        self.beta_sum = np.sum(self.beta)
        self.alpha = alpha
        self.K = len(self.alpha)
        self.W = doc.W
        self.doc = doc
        self.n_words = self.doc.n_words
        self.n_cvb_iters = n_cvb_iters
        #TODO: this probably does not work real docs because they are sentence based
        self.I_words = np.array(self.doc.I_words)
        self.all_wi_dic = {} #keys are vocab indexes and value is the list of words of that type
        self.word_counts = {} #keys are vocab indexes and value is the corresponding word count
        for i, word in enumerate(self.I_words):
            if word not in self.all_wi_dic:
                self.all_wi_dic[word] = []
                self.word_counts[word] = 0.0
            self.all_wi_dic[word].append(i)
            self.word_counts[word] += 1.0
        
        
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
            E_q_wi_k = (q_wi_k-self.gamma_q[i])*self.word_counts[self.I_words[i]] #Had word_counts_minus1 b4 but probably doesnt make sense, because I need to hide the topic assignment not the word counts
            Var_q_wi_k = E_q_wi_k*(1.0-E_q_wi_k)
            
            q_wi_k_plus_beta = q_wi_k + self.beta[self.I_words[i]]
            E_q_wi_k_plus_beta = (q_wi_k_plus_beta-self.gamma_q[i])*self.word_counts[self.I_words[i]] #Had word_counts_minus1 b4 but probably doesnt make sense, because I need to hide the topic assignment not the word counts
            
            f1 = E_q_zi_k + self.alpha
            f2 = E_q_wi_k_plus_beta
            f3 = 1.0/(E_q_zi_k+self.beta_sum)
            f4 = Var_q_zi_k/(2.0*(f1**2))
            f5 = Var_q_wi_k/(2.0*(f2**2))
            f6 = f1/(2.0*((E_q_zi_k+self.beta_sum)**2))
            
            gamma_qi = f1*f2*f3*np.exp(-f4-f5+f6)
            self.gamma_q[i] = gamma_qi/np.sum(gamma_qi)
            #print(self.gamma_q[i])
            
    def cvb_algorithm(self):
        t = trange(self.n_cvb_iters, desc='', leave=True)
        for i in t:
            t.set_description("CVB Iter %i" % (i))
            self.cvb_iter()
            
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
        
K = 3
W = 5
alpha = [15]*K
beta = [0.6]*W
n_words = 1000
n_iters = 1000

doc_synth = CVBSynDoc(alpha, beta, n_words)
vi_tt_model = TopicTrackingVIModel(alpha, beta, doc_synth, n_cvb_iters=n_iters)
vi_tt_model.cvb_algorithm()
hyp_word_topics = vi_tt_model.get_word_topics()
ref_word_topics = doc_synth.Z
print(hyp_word_topics)
print("ARI %f", adjusted_rand_score(ref_word_topics, hyp_word_topics))
        