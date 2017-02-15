'''
Created on Jan 27, 2017

@author: root
'''
import numpy as np
from scipy import sparse
from scipy.special import gammaln
from tqdm import trange
import logging
from debug.debug_tools import print_matrix_heat_map
from scipy.misc import logsumexp

class RndTopicsModel(object):
    def __init__(self, gamma, alpha, beta, K, doc, log_flag=False):
        if log_flag:
            logging.basicConfig(format='%(levelname)s:%(message)s',\
                                filename='logging/RndTopicsModel.log',\
                                filemode='w',\
                                level=logging.INFO)
        else:
            logger = logging.getLogger()
            logger.disabled = True
        
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.K = K
        self.W = doc.W
        self.doc = doc
        '''
        Array with the length of each sentence.
        Note that the length of the sentences is variable
        '''
        self.sents_len = doc.sents_len
        self.n_sents = doc.n_sents
        
        #Initializing with a random state
        self.pi = np.random.beta(gamma, gamma)
        self.rho = np.random.binomial(1, self.pi, size=doc.n_sents)
        self.rho[-1] = 0
        #Need to append last sentence, otherwise last segment wont be taken into account
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [doc.n_sents-1])
        self.n_segs = len(self.rho_eq_1)

        #Note: just to debug the initial state
        self.theta = sparse.csr_matrix((self.n_segs, K))
        self.phi = sparse.csr_matrix([np.random.dirichlet([self.beta]*self.W) for k in range(self.K)])
        #Matrix with the counts of the words in each sentence 
        self.U_W_counts = doc.U_W_counts
        #Matrix with the topics of the ith word in each u sentence 
        self.U_I_topics = sparse.csr_matrix((doc.n_sents, max(self.sents_len)))
        #Matrix with the word index of the ith word in each u sentence 
        self.U_I_words = doc.U_I_words
        #Matrix with the counts of the topic assignments in each sentence 
        self.U_K_counts = sparse.csr_matrix((doc.n_sents, self.K))
        #Matrix with the number of times each word in the vocab was assigned with topic k
        self.W_K_counts = sparse.csr_matrix((self.W, self.K))
                
        '''
        Generating all segments
        Note: Su_index is the index of the segment.
        Su_index = 0 - first segment
        Su_index = 1 - second segment
        ...
        '''
        for Su_index in range(self.n_segs):
            Su_begin, Su_end = self.get_Su_begin_end(Su_index)
            theta_Su = self.draw_theta(self.alpha)
            self.theta[Su_index, :] = theta_Su
            self.init_Z_Su(theta_Su, Su_begin, Su_end)
        
        '''
        #Note: this is just to debug without sampling rho
        self.rho = doc.rho
        self.rho_eq_1 = doc.rho_eq_1
        
        self.U_I_topics = doc.U_I_topics
        self.U_K_counts = doc.U_K_counts
        self.W_K_counts = doc.W_K_counts
        
        #Note: Making an experiment where I only sample z_ui from the first segment
        #rho and the rest of Z have the true value.
        Su_begin, Su_end = self.get_Su_begin_end(0)
        self.U_I_topics[Su_end:] = doc.U_I_topics[Su_end:]
        self.U_K_counts = sparse.csr_matrix((doc.n_sents, self.K))
        self.W_K_counts = sparse.csr_matrix((self.W, self.K))
        for u in range(self.n_sents):
            for i in range(self.sents_len[u]):
                z_ui = self.U_I_topics[u, i]
                w_ui = self.U_I_words[u, i]
                self.U_K_counts[u, z_ui] += 1
                self.W_K_counts[w_ui, z_ui] += 1
        '''
                
    def get_Su_begin_end(self, Su_index):
        Su_end = self.rho_eq_1[Su_index] + 1
        if Su_index == 0:
            Su_begin = 0
        else:
            Su_begin = self.rho_eq_1[Su_index-1] + 1
        return (Su_begin, Su_end)
        
    def draw_theta(self, alpha):
        theta = np.random.dirichlet([alpha]*self.K)
        return theta
    
    def init_Z_Su(self, theta_Su, Su_begin, Su_end):
        for u in range(Su_begin, Su_end):
            u_topic_counts = np.zeros(self.K)
            for i in range(self.sents_len[u]):
                z_u_i = np.nonzero(np.random.multinomial(1, theta_Su))[0][0]
                u_topic_counts[z_u_i] += 1.0
                self.U_I_topics[u, i] = z_u_i
                w_u_i = self.U_I_words[u, i]
                self.W_K_counts[w_u_i, z_u_i] += 1.0
            self.U_K_counts[u, :] = u_topic_counts
    
    '''
    This function gives the topic assignment z of word u,i probability
    given topic k.
    w_ui - index (in the vocab) of sentence ith word in sentence u
    k - topic
    Note: this function does not modify the w_ui counts.
    '''
    def prob_z_ui_k(self, w_ui, k, Su_index, n_Su):
        n_k_ui = self.W_K_counts[w_ui, k]
        n_t = self.W_K_counts[:, k].sum()
        f1 = (n_k_ui+self.beta)/(n_t + self.W*self.beta)
        
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        n_Su_z_ui = self.U_K_counts[Su_begin:Su_end, k].sum()
        #n_Su = np.sum(self.U_K_counts[Su_begin:Su_end, :])
        f2 = (n_Su_z_ui+self.alpha)/(n_Su + self.K*self.alpha)
        
        return f1 / f2
    
    def log_prob_z_ui_k(self, w_ui, k, Su_index, n_Su):
        n_k_ui = self.W_K_counts[w_ui, k]
        n_t = self.W_K_counts[:, k].sum()
        log_f1 = np.log(n_k_ui+self.beta) - np.log(n_t + self.W*self.beta)
        
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        n_Su_z_ui = self.U_K_counts[Su_begin:Su_end, k].sum()
        #n_Su = np.sum(self.U_K_counts[Su_begin:Su_end, :])
        log_f2 = np.log(n_Su_z_ui+self.alpha) - np.log(n_Su + self.K*self.alpha)
        
        return log_f1 + log_f2
    
    '''
    This function samples the topic assignment z of word u,i
    according to the probability of the possible topics in K.
    u - sentence number
    i - ith word from u to be sampled
    '''
    def sample_z_ui(self, u, i, Su_index, n_Su):
        '''
        Since this is for the Gibbs Sampler, we need to remove
        word w_ui from segment and topic counts
        '''
        w_ui = self.U_I_words[u, i]
        z_ui = self.U_I_topics[u, i]
        w_z_ui_count = self.W_K_counts[w_ui, z_ui]
        if w_z_ui_count > 0:
            self.W_K_counts[w_ui, z_ui] -= 1
        u_z_ui_count = self.U_K_counts[u, z_ui]
        if u_z_ui_count > 0:
            self.U_K_counts[u, z_ui] -= 1
        #self.W_K_counts[w_ui, z_ui] -= 1
        #self.U_K_counts[u, z_ui] -= 1
        
        topic_probs = []
        for k in range(self.K):
            topic_probs.append(self.prob_z_ui_k(w_ui, k, Su_index, n_Su))
        topic_probs = topic_probs / np.sum(topic_probs)
        logging.info('sample_z_ui: topic_probs %s', str(topic_probs))
        z_ui_t_plus_1 = np.nonzero(np.random.multinomial(1, topic_probs))[0][0]
        self.W_K_counts[w_ui, z_ui_t_plus_1] += 1
        self.U_K_counts[u, z_ui_t_plus_1] += 1
        self.U_I_topics[u, i] = z_ui_t_plus_1
        
    def sample_log_z_ui(self, u, i, Su_index, n_Su):
        '''
        Since this is for the Gibbs Sampler, we need to remove
        word w_ui from segment and topic counts
        '''
        w_ui = self.U_I_words[u, i]
        z_ui = self.U_I_topics[u, i]
        w_z_ui_count = self.W_K_counts[w_ui, z_ui]
        if w_z_ui_count > 0:
            self.W_K_counts[w_ui, z_ui] -= 1
        u_z_ui_count = self.U_K_counts[u, z_ui]
        if u_z_ui_count > 0:
            self.U_K_counts[u, z_ui] -= 1
        #self.W_K_counts[w_ui, z_ui] -= 1
        #self.U_K_counts[u, z_ui] -= 1
        
        topic_log_probs = []
        for k in range(self.K):
            topic_log_probs.append(self.log_prob_z_ui_k(w_ui, k, Su_index, n_Su))
        topic_probs = np.exp(topic_log_probs - np.log(np.sum(np.exp(topic_log_probs))))
        logging.info('sample_z_ui: topic_log_probs %s', str(topic_probs))
        z_ui_t_plus_1 = np.nonzero(np.random.multinomial(1, topic_probs))[0][0]
        self.W_K_counts[w_ui, z_ui_t_plus_1] += 1
        self.U_K_counts[u, z_ui_t_plus_1] += 1
        self.U_I_topics[u, i] = z_ui_t_plus_1
        
    '''
    Samples all Z variables.
    '''
    def sample_z(self):
        Su_index = 0
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        n_Su = np.sum(self.U_K_counts[Su_begin:Su_end, :])-1
        for u, rho_u in zip(range(self.n_sents), self.rho):
            for i in range(self.sents_len[u]):
                self.sample_log_z_ui(u, i, Su_index, n_Su)
            if rho_u == 1:
                #break
                Su_index += 1
                Su_begin, Su_end = self.get_Su_begin_end(Su_index)
                n_Su = np.sum(self.U_K_counts[Su_begin:Su_end, :])-1
        logging.info('sample_z:\n%s', str(self.W_K_counts.toarray()))
            
    '''
    This function assumes the states of the variables is
    already such as rho_u = 0. This is aspect is similar
    to prob_z_ui_k.
    '''    
    def log_prob_rho_u_eq_0(self, Su_begin, Su_end):
        n0 = self.n_sents - self.n_segs
        log_f1 = np.log(n0 + self.gamma) - np.log(self.n_sents + 2.0*self.gamma)
        
        #TODO: check if we should be hiding the z_u counts (I think not).
        S_u0 = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis = 0)
        
        #Note: applying log trick to gamma function 
        f2_num = (gammaln(S_u0+self.alpha)).sum()
        n_Su_0 = S_u0.sum()
        f2_dem = gammaln(n_Su_0+self.K*self.alpha)
        log_f2 = f2_num - f2_dem
        
        logging.info('log_prob_rho_u_eq_0: log_f1 %s log_f2 %s', str(log_f1), str(log_f2))
        return log_f1 + log_f2
    
    def log_prob_rho_u_eq_1(self, Su_minus_1_begin, Su_minus_1_end,\
                            Su_begin, Su_end):
        #Note: doing the log trick
        n1 = self.n_segs
        log_f1 = np.log(n1 + self.gamma) - np.log(self.n_sents + 2.0*self.gamma)
        
        log_f2 = gammaln(self.K*self.alpha) - gammaln(self.alpha)*self.K
        
        S_u1_minus_1 = np.sum(self.U_K_counts[Su_minus_1_begin:Su_minus_1_end, :], axis = 0)
        log_f3_num = gammaln(S_u1_minus_1+self.alpha).sum()
        n_Su1_minus_1 = S_u1_minus_1.sum()
        log_f3_dem = gammaln(n_Su1_minus_1+self.K*self.alpha)
        log_f3 = log_f3_num - log_f3_dem
        
        S_u1 = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis = 0)
        log_f4_num = gammaln(S_u1+self.alpha).sum()
        n_Su_1 = S_u1.sum()
        log_f4_dem = gammaln(n_Su_1+self.K*self.alpha)
        log_f4 = log_f4_num - log_f4_dem
        
        logging.info('log_prob_rho_u_eq_1: log_f1 %s log_f2 %s log_f3 %s log_f4 %s', str(log_f1), str(log_f2), str(log_f3), str(log_f4))
        return log_f1 + log_f2 + log_f3 + log_f4
    
    '''
    This function computes the begin and end of the segment
    merged at sentence u.
    
    Note: this function assumes that rho_u = 1.
    If rho_u = 0 then there would be no segments to merge
    and this method should just not be called.
    
    Note: we only worry about the begin and end of the merged
    segment because we only need this information to sample
    rho_u = 0. Its not necessary to change theta at all. 
    '''
    def merge_segments(self, Su_index):
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        Su_plus_1_begin, Su_pus_1_end = self.get_Su_begin_end(Su_index+1)
        return Su_begin, Su_pus_1_end
    
    '''
    This function splits segment Su_index at sentence u
    '''
    def split_segments(self, u, Su_index):
        begin, end = self.get_Su_begin_end(Su_index)
        Su_minus_1_begin = begin
        '''
        Note: when slicing matrix mat[0:1] give only
        the first line of the matrix. Thus, if we
        the slice with the u sentence we need to
        have the end at u + 1.         
        '''
        Su_minus_1_end = u + 1
        Su_begin = u + 1
        Su_end = end
        
        return Su_minus_1_begin, Su_minus_1_end,\
               Su_begin, Su_end
               
    def commit_merge(self, u, Su_index):
        self.rho[u] = 0
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        
    def commit_split(self, u, Su_index):
        self.n_segs += 1
        self.rho[u] = 1
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
               
    def sample_rho_u(self, u, Su_index):
        rho_u = self.rho[u]
        if rho_u == 1:
            self.n_segs -= 1
        self.n_sents -= 1
        
        if rho_u == 0:
            #Case where we do NOT need to merge segments
            Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        else:
            Su_begin, Su_end = self.merge_segments(Su_index)
        log_prob_0 = self.log_prob_rho_u_eq_0(Su_begin, Su_end)
        
        if rho_u == 1:
            #Case where we do NOT need to split segments
            Su_minus_1_begin, Su_minus_1_end = self.get_Su_begin_end(Su_index)
            Su_begin, Su_end = self.get_Su_begin_end(Su_index+1)
        else:
            Su_minus_1_begin, Su_minus_1_end, \
            Su_begin, Su_end = self.split_segments(u, Su_index)
            
        log_prob_1 = self.log_prob_rho_u_eq_1(Su_minus_1_begin, Su_minus_1_end,\
                                              Su_begin, Su_end)
        
        prob_1 = np.exp(log_prob_1 - np.logaddexp(log_prob_0, log_prob_1))
        rho_u_new = np.random.binomial(1, prob_1)
        logging.info('sample_rho_u: log_prob_0 %0.2f log_prob_1 %0.2f prob_1 %s', log_prob_0, log_prob_1, str(prob_1))
        
        #Commit the changes according to rho_u_new
        self.n_sents += 1
        
        if rho_u == rho_u_new:
            '''
            Case where we sampled the same segmentation, thus,
            we only need to restore the n_segs variable.
            '''
            if rho_u == 1:
                self.n_segs += 1
        else:
            if rho_u_new == 0:
                self.commit_merge(u, Su_index)
            else:
                self.commit_split(u, Su_index)
                
    '''
    Samples all rho variables.
    '''
    def sample_rho(self):
        Su_index = 0
        '''
        Note: the last sentence is always rho = 0
        (it cannot be a topic change since there are no more sentences)
        '''
        for u in range(self.n_sents-1):
            self.sample_rho_u(u, Su_index)
            '''
            Note: it is crucial to notice that the sampling
            of rho changes self.rho. Thus, we can only rely
            on the values after sampling to determine which
            Su_index we are at.
            '''
            if self.rho[u] == 1:
                Su_index += 1