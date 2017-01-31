'''
Created on Jan 27, 2017

@author: root
'''
import numpy as np
from scipy import sparse
from scipy.special import digamma

class TopicTrackingModel(object):
    def __init__(self, gamma, alpha, beta, K, doc):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.W = doc.W
        '''
        Array with the length of each sentence.
        Note that the length of the sentences is variable
        '''
        self.sents_len = doc.sents_len
        
        #Initializing with a random state
        self.pi = np.random.beta(gamma, gamma)
        self.rho = np.random.binomial(1, self.pi, size=doc.n_sents)
        self.rho[-1] = 0
        #Need to append last sentence, otherwise last segment wont be taken into account
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [doc.n_sents-1])
        self.n_segs = len(self.rho_eq_1)
        self.phi = sparse.csr_matrix([np.random.dirichlet([self.beta]*self.W) for k in range(self.K)])
        self.theta = sparse.csr_matrix((self.n_segs, self.K))
        #Matrix with the counts of the words in each sentence 
        self.U_W_counts = doc.U_W_counts
        #Matrix with the topics of the ith word in each u sentence 
        self.U_I_topics = doc.U_I_topics
        #Matrix with the word index of the ith word in each u sentence 
        self.U_I_words = doc.U_I_words
        #Matrix with the counts of the topic assignments in each sentence 
        self.U_K_counts = sparse.csr_matrix((doc.n_sents, self.K))
        #Matrix with the number of times each word in the vocab was assigned with topic k
        self.W_K_counts = sparse.csr_matrix((self.W, self.K))
        
        '''
        Generating first segment
        Note: for the first segment (the one outside the for loop)
        there is no t - 1, thus, I think we should not perform
        update_alpha and update_theta
        '''
        theta_S0 = np.random.dirichlet([self.alpha]*self.K)
        self.theta[0, :] = theta_S0
        self.init_Z_Su(0)
        
        '''
        Generating remaining segments
        Note: Su_index is the index of the segment.
        Su_index = 0 - first segment
        Su_index = 1 - second segment
        ...
        '''
        for Su_index in range(1, self.n_segs):
            theta_Su = self.draw_theta(Su_index)
            self.theta[Su_index, :] = theta_Su
            self.init_Z_Su(Su_index)
            self.update_alpha(Su_index)
            self.update_theta(Su_index, self.alpha)
        
    def get_Su_begin_end(self, Su_index):
        Su_end = self.rho_eq_1[Su_index] + 1
        if Su_index == 0:
            Su_begin = 0
        else:
            Su_begin = self.rho_eq_1[Su_index - 1] + 1
        return (Su_begin, Su_end)
        
    def draw_theta(self, Su_index):
        theta_t_minus_1 = self.theta[Su_index - 1, :]
        theta = np.random.dirichlet((([self.alpha]*self.K)*theta_t_minus_1.toarray())[0])
        return theta
    
    def update_theta(self, Su_index, alpha):
        theta_t_minus_1 = self.theta[Su_index - 1, :]
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        n_tk_vec = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis=0)
        n_t = np.sum(n_tk_vec)
        f1 = n_tk_vec + alpha*theta_t_minus_1
        f2 = n_t + alpha
        self.theta[Su_index, :] = f1[0] / f2
    
    def update_alpha(self, Su_index):
        theta_t_minus_1 = self.theta[Su_index - 1, :]
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        n_tk_vec = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis=0)
        n_t = np.sum(n_tk_vec)
        alpha_times_theta_t_minus_1 = self.alpha*theta_t_minus_1
        #I have no idea why I need .toarray() ...
        f1 = np.sum(theta_t_minus_1.multiply(digamma(n_tk_vec + alpha_times_theta_t_minus_1)\
                                              - digamma(alpha_times_theta_t_minus_1.toarray())))
        f2 = digamma(n_t + self.alpha) - digamma(self.alpha)
        self.alpha = self.alpha * (f1 / f2)
    
    def init_Z_Su(self, Su_index):
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        for u in range(Su_begin, Su_end):
            u_topic_counts = np.zeros(self.K)
            for i in range(self.sents_len[u]):
                z_u_i = np.nonzero(np.random.multinomial(1, self.theta[Su_index, :].toarray()[0]))[0][0]
                u_topic_counts[z_u_i] += 1.0
                self.U_I_topics[u, i] = z_u_i
                w_u_i = self.U_I_words[u, i]
                self.W_K_counts[w_u_i, z_u_i] += 1.0
            self.U_K_counts[u, :] = u_topic_counts
    
    '''
    This function samples the topic assignment z of word u,i
    u - sentence number
    i - ith word from u to be sampled
    '''
    def sample_z_ui(self, u, i):
        return
        