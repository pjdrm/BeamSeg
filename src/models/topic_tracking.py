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
        self.pi = np.random.beta(gamma, gamma)
        self.rho = np.random.binomial(1, self.pi, size=doc.n_sents)
        self.rho[-1] = 0
        #need to append last sentence, otherwise last segment wont be taken into account
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [doc.n_sents-1])
        self.n_segs = len(self.rho_eq_1)
        self.phi = sparse.csr_matrix([np.random.dirichlet([self.beta]*self.W) for k in range(self.K)])
        self.theta = sparse.csr_matrix((self.n_segs, self.K))
        #Matrix with the counts of the words in each sentence 
        self.U_W_counts = doc.U_W_counts
        #Matrix with the counts of the topic assignments in each sentence 
        self.U_K_counts = sparse.csr_matrix((doc.n_sents, self.K))
        #Matrix with the number of times each word in the vocab was assigned with topic k
        self.W_K_counts = sparse.csr_matrix((self.W, self.K))
        
        '''
        Initializing with a random state
        Note: for the first segment (the one outside the for loop)
        there is no t - 1, thus, I think we should not perform
        update_alpha and update_theta
        '''
        theta_S0 = np.random.dirichlet([self.alpha]*self.K)
        self.theta[0, :] = theta_S0
        self.init_Z_Su(0)
        for Su_index in range(1, self.n_segs):
            theta_Su = self.draw_theta(Su_index, self.alpha)
            self.theta[Su_index, :] = theta_Su
            self.init_Z_Su(Su_index)
            self.alpha = self.update_alpha(Su_index, self.alpha)
            self.theta[Su_index, :] = self.update_theta(Su_index, self.alpha)
        
    def get_Su_begin_end(self, Su_index):
        Su_end = self.rho_eq_1[Su_index] + 1
        if Su_index == 0:
            Su_begin = 0
        else:
            Su_begin = self.rho_eq_1[Su_index - 1] + 1
        return (Su_begin, Su_end)
        
    def draw_theta(self, Su_index, alpha):
        theta_t_minus_1 = self.theta[Su_index - 1, :]
        theta = np.random.dirichlet((([alpha]*self.K)*theta_t_minus_1.toarray())[0])
        return theta
    
    def update_theta(self, Su_index, alpha):
        theta_t_minus_1 = self.theta[Su_index - 1, :]
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        n_tk_vec = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis=0)
        n_t = np.sum(n_tk_vec)
        f1 = n_tk_vec + alpha*theta_t_minus_1
        f2 = n_t + alpha
        return f1[0] / f2
    
    def update_alpha(self, Su_index, alpha):
        theta_t_minus_1 = self.theta[Su_index - 1, :]
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        n_tk_vec = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis=0)
        n_t = np.sum(n_tk_vec)
        alpha_times_theta_t_minus_1 = alpha*theta_t_minus_1
        #I have no idea why I need .toarray() ...
        f1 = np.sum(theta_t_minus_1.multiply(digamma(n_tk_vec + alpha_times_theta_t_minus_1)\
                                              - digamma(alpha_times_theta_t_minus_1.toarray())))
        f2 = digamma(n_t + alpha) - digamma(alpha)
        return alpha * (f1 / f2)
    
    def init_Z_Su(self, Su_index):
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        for u in range(Su_begin, Su_end):
            u_topic_counts = np.zeros(self.K)
            for w in range(self.W):
                n_w = self.U_W_counts[u, w]
                if n_w == 0:
                    continue
                z_u_w = np.random.multinomial(n_w, self.theta[Su_index, :].toarray()[0])
                u_topic_counts += z_u_w
                self.W_K_counts[w, :] += z_u_w
            self.U_K_counts[u, :] = u_topic_counts