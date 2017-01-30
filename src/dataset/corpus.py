'''
Created on Jan 20, 2017

@author: root
'''
import numpy as np
from scipy import sparse
from scipy.special import digamma

class SyntheticDocument(object):
    def __init__(self, pi, alpha, beta, K, W, n_sents, sentence_l):
        self.alpha = alpha
        self.K = K
        self.W = W
        self.n_sents = n_sents
        self.sent_len = sentence_l
        self.rho = np.random.binomial(1, pi, size=n_sents)
        #I assume that a sentence u with rho_u = 1 belong to the previous segment.
        #rho_u = 1 means a segment is coming next, this does not make sense for 
        #the last sentence. Thus, we set it to 0.
        self.rho[-1] = 0
        #need to append last sentence, otherwise last segment wont be taken into account
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [n_sents-1])
        self.phi = sparse.csr_matrix([np.random.dirichlet([beta]*W) for k in range(K)])
        self.n_segs = len(self.rho_eq_1)
        self.theta = sparse.csr_matrix((self.n_segs, K))
        theta_S0 = np.random.dirichlet([self.alpha]*K)
        self.theta[0, :] = theta_S0
        self.U_W_counts = sparse.csr_matrix((n_sents, W))
        self.U_K_counts = sparse.csr_matrix((n_sents, K))
        
    def get_Su_begin_end(self, Su_index):
        Su_end = self.rho_eq_1[Su_index] + 1
        if Su_index == 0:
            Su_begin = 0
        else:
            Su_begin = self.rho_eq_1[Su_index - 1] + 1
        return (Su_begin, Su_end)
    
    '''
    Generating word topic assignments for the segment Su
    u - sentence index
    u_word_count - word vector representation of sentence u
    z_u_i - topic assignment of the ith word in sentence u
    '''
    def generate_Su(self, Su_index):
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        print("Generating words for %d - %d segment" % (Su_begin, Su_end))
        for u in range(Su_begin, Su_end):
            u_word_count = np.zeros(self.W)
            u_topic_counts = np.zeros(self.K)
            for word_draw in range(self.sent_len):
                z_u_i = np.nonzero(np.random.multinomial(1, self.theta[Su_index, :].toarray()[0]))[0][0]
                #print("Topic %d" % z_u_i)
                u_topic_counts[z_u_i] += 1.0
                w_u_i = np.nonzero(np.random.multinomial(1, self.phi[z_u_i].toarray()[0]))[0][0]
                u_word_count[w_u_i] += 1.0
            self.U_W_counts[u, :] = u_word_count
            self.U_K_counts[u, :] = u_topic_counts
            
    def getText(self, vocab_dic):
        print(self.rho)
        str_text = "==========\n"
        for i, rho in enumerate(self.rho):
            for j in range(self.W):
                n_words = self.U_W_counts[i, j]
                #so inificient... there must be a way to iterate just non zero entries
                if n_words == 0:
                    continue
                str_text += ((vocab_dic[j]+" ")*n_words)
            str_text += "\n"
            if rho == 1:
                str_text += "==========\n"
        str_text += "=========="
        return str_text   
    
class SyntheticTopicTrackingDoc(SyntheticDocument):
    def __init__(self, pi, alpha, beta, K, W, n_sents, sentence_l):
        SyntheticDocument.__init__(self, pi, alpha, beta, K, W, n_sents, sentence_l)
        
    def generate_doc(self):
        '''
        Generating the first segment.
        Since its the first one I don't think it makes sense to update 
        alpha and phi because there is no t - 1.
        I am not sure of tis though, it could make sense to update
        just alpha (or both) assuming phi = phi t - 1
        '''
        self.generate_Su(0)
        
        '''
        Generating remaining segments
        Note: Su is the index of the last sentence in the segment
        '''
        for Su_index in range(1, self.n_segs):
            #print("Su_index %d Su %d" % (Su_index, Su))
            theta_Su = self.generate_theta(Su_index, self.alpha)
            self.theta[Su_index, :] = theta_Su
            self.generate_Su(Su_index)
            self.alpha = self.update_alpha(Su_index, self.alpha)
            self.theta[Su_index, :] = self.update_theta(Su_index, self.alpha)
    
    def generate_theta(self, Su_index, alpha):
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
    
    '''
    Su - segment index
    '''        
    def update_alpha(self, Su_index, alpha):
        theta_t_minus_1 = self.theta[Su_index - 1, :]
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        n_tk_vec = np.sum(self.U_K_counts[Su_begin:Su_end, :], axis=0)
        n_t = np.sum(n_tk_vec)
        alpha_times_theta_t_minus_1 = alpha*theta_t_minus_1
        #I have no idea why I need .toarray() ...
        f1 = np.sum(theta_t_minus_1.multiply(digamma(n_tk_vec + alpha_times_theta_t_minus_1) - digamma(alpha_times_theta_t_minus_1.toarray())))
        f2 = digamma(n_t + alpha) - digamma(alpha)
        return alpha * (f1 / f2)
    
class SyntheticRndTopicPropsDoc(SyntheticDocument):
    def __init__(self, pi, alpha, beta, K, W, n_sents, sentence_l):
        SyntheticDocument.__init__(self, pi, alpha, beta, K, W, n_sents, sentence_l)

    def generate_doc(self):
        for Su_index in range(self.n_segs):
            #print("Su_index %d Su %d" % (Su_index, Su))
            self.theta[Su_index, :] = self.generate_theta(self.alpha)
            self.generate_Su(Su_index)
    
    def generate_theta(self, alpha):
        theta = np.random.dirichlet([alpha]*self.K)
        return theta
    