'''
Created on Nov 1, 2017

@author: pjdrm
'''
import numpy as np
from audioop import mul

class TopicTrackingVIModel(object):

    def __init__(self, gamma, alpha, beta, K, doc):
        self.beta = beta
        self.alpha = alpha
        self.bound_thresh = 0.8
        self.gamma = gamma
        self.K = K
        self.W = doc.W
        self.doc = doc

        '''
        Array with the length of each sentence.
        Note that the length of the sentences is variable
        '''
        self.sents_len = doc.sents_len
        self.n_sents = doc.n_sents
        
        '''
        Local variational parameters
        '''
        self.rho_q, self.rho_eq_1 = self.init_rho_q([1,5])
        self.theta_q = self.init_theta(0.07*self.K)
        self.z_q = self.init_z_q()
        self.pi_q = self.init_pi_q([1,5])
        
        '''
        Global parameters
        '''
        self.phi = np.array([np.random.dirichlet([self.beta]*self.W) for k in range(self.K)])

    def init_z_q(self):
        '''
        Initializes z variational parameters. Each word
        has a z_q vector with dimension of K topics. Uses
        as Dir parameters the values in theta_q of the
        corresponding segment.
        :param dir_prior:
        '''
        n_words = np.sum(self.sents_len)
        z_q = np.zeros((n_words, self.K))
        word_index = 0
        Su_index_prev = 0
        for Su_index in self.rho_eq_1:
            for word in range(np.sum(self.sents_len[Su_index_prev:Su_index])):
                z_q[word_index] = np.random.dirichlet(self.theta_q[Su_index])
                word_index += 1
            Su_index_prev = Su_index
        
        return z_q
    
    def init_rho_q(self, beta_prior):
        '''
        Initializes rho variational parameters. Its one column
        because we assume it needs to sum to 1. The value is the probability
        of rho_u being a boundary under the varaitional dist rho_q
        :param beta_prior: prior values to generate rho_q.
        '''
        rho_q = np.random.beta(beta_prior[0], beta_prior[1], (self.n_sents,1))
        rho_eq_1 = []
        for i in range(self.n_sents):
            if rho_q[i] >= self.bound_thresh:
                rho_eq_1.append(i)
        return rho_q, rho_eq_1
    
    def init_theta_q(self, dir_prior):
        '''
        Initializes theta variational parameters
        :param dir_prior: a prior to generate Dirichlet distributions. Note
        that it the prior for a Dirichlet prior itself.
        '''
        theta_q = np.zeros((self.n_sents, self.K))
        shape, scale = 2., 2.
        '''
        Just looping through boundaries because all sentences
        from the same segment share the same theta_q parameters.
        '''
        for Su_index in self.rho_eq_1:
            theta_Su_q = np.random.dirichlet(dir_prior)*np.random.gamma(shape, scale) #Dir draws give multinomial distributions, thus the random multiplier
            theta_q[Su_index, :] = theta_Su_q
            
        return theta_q
    
    def init_pi_q(self, beta_prior):
        n_docs = len(self.doc.docs_index)
        pi_q = np.zeros((n_docs, 2))
        for i in range(n_docs):
            pi_q_i = np.random.beta(beta_prior)
            multiplier = np.random.gamma(.2, .2)
            pi_q[i,:] = np.array([(1.0-pi_q_i)*multiplier, pi_q_i*multiplier])
        return pi_q       
        
        