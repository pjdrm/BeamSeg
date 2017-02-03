'''
Created on Jan 27, 2017

@author: root
'''
import numpy as np
from scipy import sparse
from scipy.special import digamma, gamma, gammaln
from scipy import exp

class TopicTrackingModel(object):
    def __init__(self, gamma, alpha, beta, K, doc):
        self.gamma = gamma
        #self.alpha = alpha
        self.beta = beta
        self.K = K
        self.W = doc.W
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
        '''
        This array has the alpha_t at the current state.
        alpha_array[0] = alpha_t_eq_0
        alpha_array[1] = alpha_t_eq_1
        ...
        TODO: need to update this array throughout the code
        '''
        self.alpha_array = np.zeros(self.n_segs)
        self.alpha_array[0] = alpha
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
        Note: for the first alpha/theta update we use the values
        obtained from the Dirichlet.
        '''
        Su_index = 0
        theta_S0 = np.random.dirichlet([self.alpha_array[Su_index]]*self.K)
        self.theta[Su_index, :] = theta_S0
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        self.init_Z_Su(theta_S0, Su_begin, Su_end)
        alpha = self.update_alpha(theta_S0, alpha, Su_begin, Su_end)
        self.alpha_array[Su_index] = alpha
        self.theta[Su_index, :] = self.update_theta(theta_S0, alpha, Su_begin, Su_end)
        
        '''
        Generating remaining segments
        Note: Su_index is the index of the segment.
        Su_index = 0 - first segment
        Su_index = 1 - second segment
        ...
        '''
        for Su_index in range(1, self.n_segs):
            Su_begin, Su_end = self.get_Su_begin_end(Su_index)
            theta_t_minus_1 = self.theta[Su_index - 1, :]
            alpha = self.alpha_array[Su_index - 1]
            theta_Su = self.draw_theta(Su_index, alpha)
            self.theta[Su_index, :] = theta_Su
            self.init_Z_Su(theta_Su, Su_begin, Su_end)
            alpha = self.update_alpha(theta_t_minus_1, alpha, Su_begin, Su_end)
            self.alpha_array[Su_index] = alpha
            self.theta[Su_index, :] = self.update_theta(theta_t_minus_1, alpha, Su_begin, Su_end)
        
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
    
    def update_theta(self, theta_t_minus_1, alpha, Su_begin, Su_end):
        n_tk_vec = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis=0)
        n_t = np.sum(n_tk_vec)
        f1 = n_tk_vec + alpha*theta_t_minus_1
        f2 = n_t + alpha
        return f1[0] / f2
    
    def update_alpha(self, theta_t_minus_1, alpha, Su_begin, Su_end):
        n_tk_vec = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis=0)
        n_t = np.sum(n_tk_vec)
        alpha_times_theta_t_minus_1 = alpha*theta_t_minus_1
        #I have no idea why I need .toarray() ...
        f1 = np.sum(theta_t_minus_1.multiply(digamma(n_tk_vec + alpha_times_theta_t_minus_1)\
                                              - digamma(alpha_times_theta_t_minus_1.toarray())))
        f2 = digamma(n_t + alpha) - digamma(alpha)
        return alpha * (f1 / f2)
    
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
    Note: I am using the equation from the Purver paper.
    Thus, there is no - 1. The equations in my derivation have - 1 though.
    '''
    def prob_z_ui_k(self, w_ui, k, Su_index):
        n_k_ui = self.W_K_counts[w_ui, k]
        n_t = self.W_K_counts[:, k].sum()
        f1 = (n_k_ui+self.beta)/(n_t + self.W*self.beta)
        
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        n_Su_z_ui = self.U_K_counts[Su_begin:Su_end, k].sum()
        n_Su = self.U_K_counts[Su_begin:Su_end, k].sum()
        #TODO: need to take care of the Su_index = 0 case
        theta_Su_k_t_minus_1 = self.theta[Su_index - 1, k]
        f2 = (n_Su_z_ui+theta_Su_k_t_minus_1*self.alpha)/(n_Su + self.K*self.alpha)
        
        return f1 / f2
    
    '''
    This function samples the topic assignment z of word u,i
    according to the probability of the possible topics in K.
    u - sentence number
    i - ith word from u to be sampled
    TODO: check if I should be doing an update on alpha and theta
    after the new z assignment
    '''
    def sample_z_ui(self, u, i, Su_index):
        '''
        Since this is for the Gibbs Sampler, we need to remove
        word w_ui from segment and topic counts
        '''
        w_ui = self.U_I_words[u, i]
        z_ui = self.U_I_topics[u, i]
        self.W_K_counts[w_ui, z_ui] -= 1
        self.U_K_counts[u, z_ui] -= 1
        
        topic_probs = []
        for k in range(self.K):
            topic_probs.append(self.prob_z_ui_k(w_ui, k, Su_index))
        topic_probs = topic_probs / np.sum(topic_probs)
        z_ui_t_plus_1 = np.nonzero(np.random.multinomial(1, topic_probs))[0][0]
        self.W_K_counts[w_ui, z_ui_t_plus_1] += 1
        self.U_K_counts[u, z_ui_t_plus_1] += 1
    
    '''
    This function assumes the states of the variables is
    already such as rho_u = 0. This is aspect is similar
    to prob_z_ui_k.
    '''    
    def log_prob_rho_u_eq_0(self, Su_begin, Su_end, Su_index):
        n0 = self.n_sents - self.n_segs
        log_f1 = np.log(n0 + self.gamma) - np.log(self.n_sents + 2.0*self.gamma)
        
        #TODO: check if we should be hiding the z_u counts (I think not).
        S_u0 = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis = 0)
        alpha = self.alpha_array[Su_index-1]
        theta_Su_t_minus_1 = self.theta[Su_index - 1]
        #Note: applying log trick to gamma function 
        f2_num = (gammaln(S_u0+theta_Su_t_minus_1*alpha)).sum()
        n_Su_0 = S_u0.sum()
        f2_dem = gammaln(n_Su_0+self.K*alpha)
        log_f2 = f2_num - f2_dem
        
        return log_f1 + log_f2
    
    def log_prob_rho_u_eq_1(self, Su_minus_1_begin, Su_minus_1_end, alpha_Su_t_minus_1, theta_Su_t_minus_1,\
                              Su_begin, Su_end, theta_Su, alpha_Su):
        #Note: doing the log trick
        n1 = self.n_segs
        log_f1 = np.log(n1 + self.gamma) - np.log(self.n_sents + 2.0*self.gamma)
        
        log_f2 = gammaln(self.K*alpha_Su) - gammaln(alpha_Su*theta_Su).sum()
        
        #TODO: deal with the Su_index = 0 case
        S_u1_minus_1 = np.sum(self.U_K_counts[Su_minus_1_begin:Su_minus_1_end, :], axis = 0)
        log_f3_num = gammaln(S_u1_minus_1+theta_Su_t_minus_1*alpha_Su_t_minus_1).sum()
        n_Su1_minus_1 = S_u1_minus_1.sum()
        log_f3_dem = gammaln(n_Su1_minus_1+self.K*alpha_Su_t_minus_1)
        log_f3 = log_f3_num - log_f3_dem
        
        S_u1 = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis = 0)
        #TODO: probably I will have to use the log trick here
        log_f4_num = gammaln(S_u1+theta_Su*alpha_Su).sum()
        n_Su_1 = S_u1.sum()
        log_f4_dem = gammaln(n_Su_1+self.K*alpha_Su)
        log_f4 = log_f4_num - log_f4_dem
        
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
        Su_minus_1_end = u
        Su_begin = u + 1
        Su_end = end
        
        theta_t_minus_1 = self.theta[Su_index - 1, :]
        alpha_Su_t_minus_1 = self.update_alpha(theta_t_minus_1,\
                                               self.alpha_array[Su_index - 1],\
                                               Su_minus_1_begin, Su_minus_1_end)
        theta_Su_t_minus_1 = self.update_theta(theta_t_minus_1, alpha_Su_t_minus_1,\
                                               Su_minus_1_begin, Su_minus_1_end)
        
        alpha_Su = self.update_alpha(theta_Su_t_minus_1,\
                                     alpha_Su_t_minus_1,\
                                     Su_begin, Su_end)
        theta_Su = self.update_theta(theta_Su_t_minus_1, alpha_Su,\
                                     Su_begin, Su_end)
        
        return Su_minus_1_begin, Su_minus_1_end, theta_Su_t_minus_1, alpha_Su_t_minus_1,\
               Su_begin, Su_end, theta_Su, alpha_Su
               
    def commit_merge(self, u, Su_index):
        self.rho[u] = 0
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        #Note: I don't think I need to reshape here
        #self.reshape_alpha_theta_update(Su_index)
        
    def commit_split(self, u, Su_index):
        self.n_segs += 1
        self.rho[u] = 1
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        self.reshape_alpha_theta_update(Su_index)       
        
    '''
    This method is called when split/merge of segments occur
    during the sampling of rho_u.
    
    I am assuming I only need to worry about alpha/theta values
    up to segment Su_index, since rho_u will keep being sampled
    and possibly new splits and merges can occur. Thus, I just 
    reshape alpha and theta (I know when this method is called the
    number of segments has changed) and copy the values up to Su_index
    (exclusive) and calculate alpha and theta only for Su_index only.
    Thus, the segments after Su_index have alpha and theta equal to 0.
    '''
    def reshape_alpha_theta_update(self, Su_index):
        new_alpha_array = np.zeros(self.n_segs)
        new_alpha_array[0:Su_index-1] = self.alpha_array[0:Su_index-1]
        
        new_theta_mat = sparse.csr_matrix((self.n_segs, self.K))
        new_theta_mat[0:Su_index-1, :] = self.theta[0:Su_index-1, :]
        
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        theta_t_minus_1 = self.new_theta_mat[Su_index-1, :]
        alpha = self.update_alpha(theta_t_minus_1,\
                                  new_alpha_array[Su_index-1],\
                                  Su_begin, Su_end)
        new_alpha_array[Su_index] = alpha
        new_theta_mat[Su_index, :] = self.update_theta(theta_t_minus_1, alpha, Su_begin, Su_end)
            
        self.alpha_array = new_alpha_array
        self.theta = new_theta_mat
               
    def sample_rho_u(self, u, Su_index):
        rho_u = self.rho[u]
        if rho_u == 1:
            self.n_segs -= 1
        self.n_sents -= 1
        
        if rho_u == 0:
            #Case where we do not need to merge segments
            Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        else:
            Su_begin, Su_end = self.merge_segments(Su_index)
        log_prob_0 = self.log_prob_rho_u_eq_0(Su_begin, Su_end, Su_index)
        
        if rho_u == 1:
            #Case where we do not need to split segments
            Su_minus_1_begin, Su_minus_1_end = self.get_Su_begin_end(Su_index)
            alpha_Su_t_minus_1 = self.alpha_array[Su_index]
            theta_Su_t_minus_1 = self.theta[Su_index, :].toarray()
            Su_begin, Su_end = self.get_Su_begin_end(Su_index + 1)
            theta_Su = self.theta[Su_index + 1, :].toarray()
            alpha_Su = self.alpha_array[Su_index + 1]
        else:
            Su_minus_1_begin, Su_minus_1_end, alpha_Su_t_minus_1, theta_Su_t_minus_1,\
            Su_begin, Su_end, theta_Su, alpha_Su = self.split_segments(u, Su_index)
            
        log_prob_1 = self.log_prob_rho_u_eq_1(Su_minus_1_begin, Su_minus_1_end, alpha_Su_t_minus_1, theta_Su_t_minus_1,\
                                      Su_begin, Su_end, theta_Su, alpha_Su)
        
        prob_1 = np.exp(log_prob_1 - np.logaddexp(log_prob_0, log_prob_1))
        rho_u_new = np.random.binomial(1, prob_1)
        
        #Commit the changes according to rho_u_new
        self.n_sents += 1
        
        if rho_u == rho_u_new:
            '''
            Case where we sampled the same segmentation, thus,
            we only need to restore the n_segs variable.
            '''
            if rho_u == 1:
                self.n_segs += 1
                '''
                Note: it is crucial to update alpha/theta when we pass
                a segment and we are going to split/merge because
                the actual split/merge operations zero alpha/theta
                for the following segments.
                '''
                Su_begin, Su_end = self.get_Su_begin_end(Su_index)
                theta_t_minus_1 = self.theta[Su_index-1, :]
                alpha = self.update_alpha(theta_t_minus_1,\
                                          self.alpha_array[Su_index-1],\
                                          Su_begin, Su_end)
                self.alpha_array[Su_index] = alpha
                self.theta[Su_index, :] = self.update_theta(theta_t_minus_1, alpha, Su_begin, Su_end)
        else:
            if rho_u == 0:
                self.commit_merge(u, Su_index)
            else:
                self.commit_split(u, Su_index)
