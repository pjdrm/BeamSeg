'''
Created on Jan 27, 2017

Implementation of Mathew Purver's paper
"Unsupervised Topic Modelling for Multi-Party Spoken Discourse"

@author: pjdrm
'''
import numpy as np
from scipy.special import gammaln
from scipy.misc import logsumexp
from debug import log_tools
from random import shuffle
import copy

class RndTopicsModel(object):
    def __init__(self, configs, data,\
                 log_flag=False,\
                 sampler_log_file = "RndTopicsModel.log"):
        
        self.rt_seg_log = log_tools.log_init(sampler_log_file)
        if not log_flag:
            self.rt_seg_log.disabled = True
        
        self.alpha = configs["model"]["alpha"]
        self.beta = configs["model"]["alpha"]
        self.gamma = configs["model"]["gamma"]
        self.K = configs["model"]["K"]
        self.W = data.W
        self.doc = data
        '''
        Array with the length of each sentence.
        Note that the length of the sentences is variable
        '''
        self.sents_len = data.sents_len
        self.n_sents = data.n_sents
        
        #Initializing with a random state
        if configs["model"]["pi"] == "None":
            self.pi = np.random.beta(self.gamma, self.gamma)
        else:
            self.pi = configs["model"]["pi"]
        self.rho = np.random.binomial(1, self.pi, size=data.n_sents)
        self.rho[-1] = 0
        #Need to append last sentence, otherwise last segment wont be taken into account
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [data.n_sents-1])
        self.n_segs = len(self.rho_eq_1)

        #Note: just to debug the initial state
        self.theta = np.zeros((self.n_segs, self.K))
        self.phi = np.array([np.random.dirichlet([self.beta]*self.W) for k in range(self.K)])
        #Matrix with the counts of the words in each sentence 
        self.U_W_counts = data.U_W_counts
        #Matrix with the topics of the ith word in each u sentence 
        self.U_I_topics = np.zeros((data.n_sents, max(self.sents_len)))#self.data.U_I_topics
        #Matrix with the word index of the ith word in each u sentence 
        self.U_I_words = data.U_I_words
        #Matrix with the counts of the topic assignments in each sentence 
        self.U_K_counts = np.zeros((data.n_sents, self.K))#self.data.U_K_counts
        #Matrix with the number of times each word in the vocab was assigned with topic k
        self.W_K_counts = np.zeros((self.W, self.K)) #self.data.W_K_counts
                
        '''
        Generating all segments
        Note: Su_index is the index of the segment.
        Su_index = 0 - first segment
        Su_index = 1 - second segment
        ...
        '''
        for Su_index in range(self.n_segs):
            Su_begin, Su_end = self.get_Su_begin_end(Su_index, self.rho_eq_1)
            theta_Su = self.draw_theta(self.alpha)
            self.theta[Su_index, :] = theta_Su
            self.init_Z_Su(theta_Su, Su_begin, Su_end)
                
    def get_Su_begin_end(self, Su_index, rho_eq_1):
        Su_end = rho_eq_1[Su_index] + 1
        if Su_index == 0:
            Su_begin = 0
        else:
            Su_begin = rho_eq_1[Su_index-1] + 1
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
    def log_prob_z_ui_k(self, w_ui, k, Su_index, n_Su):
        n_k_ui = self.W_K_counts[w_ui, k]
        n_t = self.W_K_counts[:, k].sum()
        log_f1 = np.log(n_k_ui+self.beta) - np.log(n_t + self.W*self.beta)
        
        Su_begin, Su_end = self.get_Su_begin_end(Su_index, self.rho_eq_1)
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
    def sample_log_z_ui(self, u, i, w_ui, Su_index, topic_log_probs):
        z_ui_t_plus_1 = np.nonzero(np.random.multinomial(1, topic_log_probs))[0][0]
        self.W_K_counts[w_ui, z_ui_t_plus_1] += 1
        self.U_K_counts[u, z_ui_t_plus_1] += 1
        self.U_I_topics[u, i] = z_ui_t_plus_1
        return z_ui_t_plus_1
    
    '''
    Calculates the topic proportions for word w_ui
    '''
    def log_prob_Z(self, u, i, w_ui, z_ui, Su_index, n_Su):
        '''
        Since this is for the Gibbs Sampler, we need to remove
        word w_ui from segment and topic counts
        '''
        w_z_ui_count = self.W_K_counts[w_ui, z_ui]
        if w_z_ui_count > 0:
            self.W_K_counts[w_ui, z_ui] -= 1
        u_z_ui_count = self.U_K_counts[u, z_ui]
        if u_z_ui_count > 0:
            self.U_K_counts[u, z_ui] -= 1
        
        topic_log_probs = []
        for k in range(self.K):
            topic_log_probs.append(self.log_prob_z_ui_k(w_ui, k, Su_index, n_Su))
        topic_log_probs = np.exp(topic_log_probs - np.log(np.sum(np.exp(topic_log_probs))))
        #self.rt_seg_log.info('sample_z_ui: topic_log_probs %s', str(topic_log_probs))
        return topic_log_probs
        
    '''
    Samples all Z variables.
    '''
    def sample_z(self):
        Su_index = 0
        Su_begin, Su_end = self.get_Su_begin_end(Su_index, self.rho_eq_1)
        n_Su = np.sum(self.U_K_counts[Su_begin:Su_end, :])-1
        for u, rho_u in zip(range(self.n_sents), self.rho):
            for i in range(self.sents_len[u]):
                w_ui = self.U_I_words[u, i]
                z_ui = int(self.U_I_topics[u, i])
                topic_log_probs = self.log_prob_Z(u, i, w_ui, z_ui, Su_index, n_Su)
                self.sample_log_z_ui(u, i, w_ui, Su_index, topic_log_probs)
            if rho_u == 1:
                Su_index += 1
                Su_begin, Su_end = self.get_Su_begin_end(Su_index, self.rho_eq_1)
                n_Su = np.sum(self.U_K_counts[Su_begin:Su_end, :])-1
        #self.rt_seg_log.info('sample_z:\n%s', str(self.W_K_counts.toarray()))
            
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
        
        #self.rt_seg_log.info('log_prob_rho_u_eq_0: log_f1 %s log_f2 %s', str(log_f1), str(log_f2))
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
        
        #self.rt_seg_log.info('log_prob_rho_u_eq_1: log_f1 %s log_f2 %s log_f3 %s log_f4 %s', str(log_f1), str(log_f2), str(log_f3), str(log_f4))
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
        Su_begin, Su_end = self.get_Su_begin_end(Su_index, self.rho_eq_1)
        Su_plus_1_begin, Su_pus_1_end = self.get_Su_begin_end(Su_index+1, self.rho_eq_1)
        return Su_begin, Su_pus_1_end
    
    '''
    This function splits segment Su_index at sentence u
    '''
    def split_segments(self, u, Su_index):
        begin, end = self.get_Su_begin_end(Su_index, self.rho_eq_1)
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
               
    def merge_log_prob(self, rho_u, Su_index):
        if rho_u == 0:
            #Case where we do NOT need to merge segments
            Su_begin, Su_end = self.get_Su_begin_end(Su_index, self.rho_eq_1)
        else:
            Su_begin, Su_end = self.merge_segments(Su_index)
        log_prob_0 = self.log_prob_rho_u_eq_0(Su_begin, Su_end)
        return log_prob_0
    
    def split_log_prob(self, u, rho_u, Su_index):
        if rho_u == 1:
            #Case where we do NOT need to split segments
            Su_minus_1_begin, Su_minus_1_end = self.get_Su_begin_end(Su_index, self.rho_eq_1)
            Su_begin, Su_end = self.get_Su_begin_end(Su_index+1, self.rho_eq_1)
        else:
            Su_minus_1_begin, Su_minus_1_end, \
            Su_begin, Su_end = self.split_segments(u, Su_index)
            
        log_prob_1 = self.log_prob_rho_u_eq_1(Su_minus_1_begin, Su_minus_1_end,\
                                              Su_begin, Su_end)
        return log_prob_1
    
    def commit_merge(self, u, Su_index):
        self.rho[u] = 0
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        
    def commit_split(self, u, Su_index):
        self.n_segs += 1
        self.rho[u] = 1
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
                   
    def sample_rho_u(self, u, Su_index, rho_u, log_prob_0, log_prob_1):
        prob_1 = np.exp(log_prob_1 - np.logaddexp(log_prob_0, log_prob_1))
        self.rt_seg_log.info('sample_rho_u: u %d log_prob_0 %0.2f log_prob_1 %0.2f prob_1 %s', u, log_prob_0, log_prob_1, str(prob_1))
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
        else:
            if rho_u_new == 0:
                self.commit_merge(u, Su_index)
            else:
                self.commit_split(u, Su_index)
        return rho_u_new
                
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
            rho_u = self.rho[u]
            if rho_u == 1:
                self.n_segs -= 1
            self.n_sents -= 1
            
            log_prob_0 = self.merge_log_prob(rho_u, Su_index)
            log_prob_1 = self.split_log_prob(u, rho_u, Su_index)
            self.sample_rho_u(u, Su_index, rho_u, log_prob_0, log_prob_1)
            '''
            Note: it is crucial to notice that the sampling
            of rho changes self.rho. Thus, we can only rely
            on the values after sampling to determine which
            Su_index we are at.
            '''
            if self.rho[u] == 1:
                Su_index += 1

    def log_prob_joint_dist(self, gamma, alpha, alpha, rho_eq_1, W_K_counts, U_K_counts):
        n_sents = U_K_counts.shape[0]
        K = U_K_counts.shape[1]
        n1 = len(rho_eq_1)
        n0 = n_sents - n1
        W = W_K_counts.shape[0]
        
        np_W_K_counts = W_K_counts
        
        log_p_c_f1 = gammaln(2.0*gamma) - (gammaln(gamma)*2)
        log_p_c_f2 = (gammaln(n1+gamma) + gammaln(n0+gamma)) - gammaln(n_sents+2.0*gamma)
        log_p_c = log_p_c_f1 + log_p_c_f2
        
        log_p_wz_f1 = (gammaln(W*alpha) - gammaln(alpha)*W)*K
        log_p_wz_f2 = (np.sum(gammaln(np_W_K_counts + alpha), axis = 0) - gammaln(np.sum(np_W_K_counts, axis=0) + W*alpha)).sum()
        log_p_wz = log_p_wz_f1 + log_p_wz_f2
        
        log_p_zc_f1 = (gammaln(K*alpha) - gammaln(alpha)*K)*n1
        log_p_zc_f2 = 0.0
        for Su_index in range(n1):
            Su_begin, Su_end = self.get_Su_begin_end(Su_index, rho_eq_1)
            Su_K_counts = U_K_counts[Su_begin:Su_end, :]
            n_K_Su = np.sum(Su_K_counts, axis = 0)
            deno = (gammaln(n_K_Su + alpha)).sum()
            n_Su = n_K_Su.sum()
            num = gammaln(n_Su+ K*alpha)
            log_p_zc_f2 += deno -num
        log_p_zc = log_p_zc_f1 + log_p_zc_f2
        
        return log_p_c + log_p_wz + log_p_zc

'''
Efficient version, based on a caching scheme, of the
RndTopicsModel class.
'''    
class RndTopicsCacheModel(RndTopicsModel):
    def __init__(self, configs, data,\
                 log_flag=False,\
                 sampler_log_file = "RndTopicsModel.log"):
        RndTopicsModel.__init__(self, configs, data,\
                                log_flag,\
                                sampler_log_file)
        '''
        This is the main trick to sample Z efficiently.
        The idea is to sample by word type (by the order
        they occur in the document). A cache hit will occur
        when 2 words of the same type, in the same segment,
        have the same topic. In these conditions, I just need
        to cache the topic probabilities of the previous order.
        '''
        self.sample_order = self.calc_sample_order()
        
    def calc_sample_order(self):
        sample_order_dic = {}
        for u in range(self.n_sents):
            for i in range(self.sents_len[u]):
                w_ui = self.U_I_words[u,i]
                if w_ui not in sample_order_dic:
                    sample_order_dic[w_ui] = []
                sample_order_dic[w_ui].append((u,i, w_ui))
        res = []
        for w_ui in sample_order_dic:
            res += sample_order_dic[w_ui]
        return res           
                
    def sample_z(self):
        z_ui_prev = -1
        w_ui_prev = -1
        Su_index_prev = -1
        cache_topic_log_probs = -1
        n_Su_array = self.calc_n_Su_array()
        for u, i, w_ui in self.sample_order:
            Su_index = self.get_Su_index(u)
            n_Su = n_Su_array[Su_index]
            w_ui = self.U_I_words[u, i]
            z_ui = int(self.U_I_topics[u,i])
            
            if  z_ui == z_ui_prev and\
                w_ui == w_ui_prev and\
                Su_index_prev == Su_index:
                self.W_K_counts[w_ui, z_ui] -= 1
                self.U_K_counts[u, z_ui] -= 1
                z_ui_new = self.sample_log_z_ui(u, i, w_ui, Su_index, cache_topic_log_probs)
            else:
                topic_log_probs = self.log_prob_Z(u, i, w_ui, z_ui, Su_index, n_Su)
                cache_topic_log_probs = topic_log_probs
                z_ui_new = self.sample_log_z_ui(u, i, w_ui, Su_index, topic_log_probs)
            z_ui_prev = z_ui_new
            w_ui_prev = w_ui
            Su_index_prev = Su_index
            
    '''
    The caching scheme for sampling rho consist of
    taking advantage of the fact that sentences in the same
    segment have the same merge probability.
    '''        
    def sample_rho(self, doc_index = None):
        cacheFlag = False
        cache_log_prob_0 = -1
        if doc_index is not None:
            sents = range(doc_index[0], doc_index[1])
            Su_index = self.get_Su_index(doc_index[0])
        else:
            sents = range(self.n_sents-1)
            Su_index = 0
            
        '''
        Note: the last sentence is always rho = 0
        (it cannot be a topic change since there are no more sentences)
        '''
        for u in sents:
            rho_u = self.rho[u]
            if rho_u == 1:
                self.n_segs -= 1
            self.n_sents -= 1
            
            '''
            Note: we cannot use the cache value if rho == 1,
            because the probability of a merge is different from what 
            we have in cache.
            '''
            if rho_u == 0 and cacheFlag:
                log_prob_0 = cache_log_prob_0
            else:
                log_prob_0 = self.merge_log_prob(rho_u, Su_index)
                cache_log_prob_0 = log_prob_0
            log_prob_1 = self.split_log_prob(u, rho_u, Su_index)
            rho_u_new = self.sample_rho_u(u, Su_index, rho_u, log_prob_0, log_prob_1)
            '''
            Note: it is crucial to notice that the sampling
            of rho changes self.rho. Thus, we can only rely
            on the values after sampling to determine which
            Su_index we are at.
            '''
            if rho_u_new == 1:
                Su_index += 1
                cacheFlag = False
            else:
                cacheFlag = True
        return self.rho
        
    def calc_n_Su_array(self):
        n_Su_array = np.zeros(self.n_segs)
        for Su_index in range(self.n_segs):
            Su_begin, Su_end = self.get_Su_begin_end(Su_index, self.rho_eq_1)
            n_Su = np.sum(self.U_K_counts[Su_begin:Su_end, :])-1
            if n_Su == -1.0:
                n_Su = 0.0
            n_Su_array[Su_index] = n_Su
        return n_Su_array
    
    def get_Su_index(self, u):
        for Su_index, rho1 in enumerate(self.rho_eq_1):
            if u <= rho1:
                return Su_index

class RndScanOrderModel(RndTopicsModel):
    def __init__(self, configs, data,\
                 log_flag=False,\
                 sampler_log_file = "RndTopicsModel.log"):
        RndTopicsModel.__init__(self, configs, data,\
                                log_flag,\
                                sampler_log_file)
        
        self.z_sample_order = self.calc_sample_order()
        self.rho_sample_order = list(range(0, self.n_sents))
        shuffle(self.rho_sample_order)
    
    '''
    Returns a z sample order by word type
    and order of occurrence.
    '''    
    def calc_sample_order(self):
        sample_order_dic = {}
        for u in range(self.n_sents):
            for i in range(self.sents_len[u]):
                w_ui = self.U_I_words[u,i]
                if w_ui not in sample_order_dic:
                    sample_order_dic[w_ui] = []
                sample_order_dic[w_ui].append((u,i, w_ui))
        res = []
        for w_ui in sample_order_dic:
            res += sample_order_dic[w_ui]
        shuffle(res)
        return res
    
    def sample_z(self):
        z_order = copy.deepcopy(self.z_sample_order)
        c_order = copy.deepcopy(self.rho_sample_order)
        while len(z_order) > 0 and len(c_order) > 0:
            var_typ = np.random.binomial(1, 0.5)
            if var_typ == 1:
                u, i, w_ui = z_order.pop(0)
                self.sample_z_new(u, i, w_ui)
            else:
                self.sample_rho_new(c_order.pop(0))
        
        while len(z_order) > 0:
            u, i, w_ui = z_order.pop(0)
            self.sample_z_new(u, i, w_ui)
            
        while len(c_order) > 0: 
            self.sample_rho_new(c_order.pop(0))
        shuffle(self.z_sample_order)
        shuffle(self.rho_sample_order)
        
    def sample_rho(self, doc_index = None):
            return self.rho
                
    def sample_z_new(self, u, i, w_ui):
        Su_index = self.get_Su_index(u)
        Su_begin, Su_end = self.get_Su_begin_end(Su_index, self.rho_eq_1)
        n_Su = np.sum(self.U_K_counts[Su_begin:Su_end, :])-1
        if n_Su == -1.0:
            n_Su = 0.0
        w_ui = self.U_I_words[u, i]
        z_ui = int(self.U_I_topics[u,i])
        topic_log_probs = self.log_prob_Z(u, i, w_ui, z_ui, Su_index, n_Su)
        self.sample_log_z_ui(u, i, w_ui, Su_index, topic_log_probs)
            
    '''
    The caching scheme for sampling rho consist of
    taking advantage of the fact that sentences in the same
    segment have the same merge probability.
    '''        
    def sample_rho_new(self, u):
        Su_index = self.get_Su_index(u)
        rho_u = self.rho[u]
        if rho_u == 1:
            self.n_segs -= 1
        self.n_sents -= 1
        
        log_prob_0 = self.merge_log_prob(rho_u, Su_index)
        log_prob_1 = self.split_log_prob(u, rho_u, Su_index)
        self.sample_rho_u(u, Su_index, rho_u, log_prob_0, log_prob_1)
    
    def get_Su_index(self, u):
        for Su_index, rho1 in enumerate(self.rho_eq_1):
            if u <= rho1:
                return Su_index
        
U_K_counts_g = None
U_I_topics_g = None
W_K_counts_g = None

class RndTopicsParallelModel(RndTopicsModel):
    def __init__(self, gamma, alpha, alpha, K, data,\
                 log_flag=False,\
                 sampler_log_file = "RndTopicsModel.log"):
        RndTopicsModel.__init__(self, gamma, alpha, alpha, K, data,\
                                    log_flag,\
                                    sampler_log_file)
        self.log_flag = log_flag
        self.segmentors = self.get_segmentors()
        self.sample_rho_args = []
        for doc_i in range(self.doc.n_docs):
            self.sample_rho_args.append((self.segmentors[doc_i],))
        
        self.U_K_counts = self.base_seg.U_K_counts
        self.U_I_topics = self.base_seg.U_I_topics
        self.W_K_counts = self.base_seg.W_K_counts
        
    def get_segmentors(self):
        self.base_seg = RndTopicsCacheModel(self.gamma, self.alpha,\
                                       self.beta, self.K,\
                                       self.doc, self.log_flag)
        segmentors = []
        doc_begin = 0
        for doc_end in self.doc.docs_index:
            seg = RndTopicsCacheModel(self.gamma, self.alpha,\
                                      self.beta, self.K,\
                                      self.doc, self.log_flag)
            
            #Note: it's crucial that each seg as its own rho, n_segs and n_sents
            seg.n_sents = doc_end - doc_begin
            seg.rho = np.array(self.base_seg.rho[doc_begin:doc_end])
            seg.rho_eq_1 = np.append(np.nonzero(seg.rho)[0], [seg.n_sents-1])
            seg.n_segs = len(seg.rho_eq_1)
        
            '''
            The idea to share the matrix so that by sampling Z 
            in on seg all the other get updated too.
            '''
            seg.U_K_counts = self.base_seg.U_K_counts[doc_begin:doc_end, :]
            seg.U_I_topics = self.base_seg.U_I_topics[doc_begin:doc_end, :]
            seg.W_K_counts = self.base_seg.W_K_counts[doc_begin:doc_end, :]
            segmentors.append(seg)
            doc_begin = doc_end
        return segmentors
    
    def sample_z(self):
        self.base_seg.sample_z()
        
    '''
    Note: implementation of parallelization is harder than I thought.
    This is due to shared variables like rho_eq1.
    '''
    def sample_rho(self):
        #thread_pool = mp.ProcessingPool(3)
        results = []#thread_pool.map(self.sample_rho_parallel, range(self.doc.n_docs))
        for doc_i in range(self.doc.n_docs):
            results.append(self.segmentors[doc_i].sample_rho())
        #Base seg needs to have rho variables correct to sample Z
        self.base_seg.rho = []
        self.base_seg.rho_eq_1 = []
        for doc_i, rho in enumerate(results):
            seg_i = self.segmentors[doc_i]
            seg_i.rho = rho
            seg_i.rho_eq_1 = np.append(np.nonzero(seg_i.rho)[0], [self.base_seg.n_sents-1])
            seg_i.n_segs = len(seg_i.rho_eq_1)
            self.base_seg.rho = np.concatenate([self.base_seg.rho, rho])
        self.base_seg.rho_eq_1 = np.append(np.nonzero(self.base_seg.rho)[0], [self.base_seg.n_sents-1])
        self.base_seg.n_segs = len(self.base_seg.rho_eq_1)
        self.rho = self.base_seg.rho
        self.rho_eq_1 = self.base_seg.rho_eq_1
        self.n_segs = self.base_seg.n_segs
        
    def sample_rho_parallel(self, doc_i):
        return self.segmentors[doc_i].sample_rho()