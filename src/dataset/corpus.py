'''
Created on Jan 20, 2017

@author: root
'''
import numpy as np
from scipy import sparse
from scipy import int8
from model.topic_tracking import TopicTrackingModel

class SyntheticDocument(object):
    def __init__(self, pi, alpha, beta, K, W, n_sents, sent_len):
        self.alpha = alpha
        self.K = K
        self.W = W
        self.n_sents = n_sents
        self.sent_len = sent_len
        self.sents_len = [sent_len]*n_sents
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
        self.U_W_counts = sparse.csr_matrix((n_sents, W), dtype=int8)
        self.U_K_counts = sparse.csr_matrix((n_sents, K), dtype=int8)
        self.U_I_topics = sparse.csr_matrix((n_sents, sent_len), dtype=int8)
        self.U_I_words = sparse.csr_matrix((n_sents, sent_len), dtype=int8)
    
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
                u_topic_counts[z_u_i] += 1
                w_u_i = np.nonzero(np.random.multinomial(1, self.phi[z_u_i].toarray()[0]))[0][0]
                u_word_count[w_u_i] += 1
                self.U_I_topics[u, word_draw] = z_u_i
                self.U_I_words[u, word_draw] = w_u_i
            self.U_W_counts[u, :] = u_word_count
            self.U_K_counts[u, :] = u_topic_counts
            
    def getText(self, vocab_dic):
        str_text = "==========\n"
        for i, rho in enumerate(self.rho):
            for j in range(self.sents_len[i]):
                w_ij = self.U_I_words[i, j]
                str_text += vocab_dic[w_ij] + " "
            str_text += "\n"
            if rho == 1:
                str_text += "==========\n"
        str_text += "=========="
        return str_text   
    
'''
Making this class to inherit from TopicTrackingModel
to use the draw_theta, update_alpha, and update_theta.
Otherwise I would have to repeat the code.
'''
class SyntheticTopicTrackingDoc(SyntheticDocument, TopicTrackingModel):
    def __init__(self, pi, alpha, beta, K, W, n_sents, sentence_l):
        SyntheticDocument.__init__(self, pi, alpha, beta, K, W, n_sents, sentence_l)
        
    def generate_doc(self):
        '''
        Generating the first segment.
        Assumes theta t - 1 is the draw from the
        Dirichlet in the beginning.
        '''
        Su_index = 0
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        theta_S0 = self.theta[Su_index, :]
        self.generate_Su(Su_index)
        self.alpha = self.update_alpha(theta_S0, self.alpha, Su_begin, Su_end)
        self.theta[Su_index, :] = self.update_theta(theta_S0, self.alpha, Su_begin, Su_end)
        
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
            theta_Su = self.draw_theta(Su_index, self.alpha)
            self.theta[Su_index, :] = theta_Su
            self.generate_Su(Su_index)
            self.alpha = self.update_alpha(theta_t_minus_1, self.alpha, Su_begin, Su_end)
            self.theta[Su_index, :] = self.update_theta(theta_t_minus_1, self.alpha, Su_begin, Su_end)
    
class SyntheticRndTopicPropsDoc(SyntheticDocument):
    def __init__(self, pi, alpha, beta, K, W, n_sents, sentence_l):
        SyntheticDocument.__init__(self, pi, alpha, beta, K, W, n_sents, sentence_l)

    def generate_doc(self):
        for Su_index in range(self.n_segs):
            #print("Su_index %d Su %d" % (Su_index, Su))
            self.theta[Su_index, :] = self.draw_theta(self.alpha)
            self.generate_Su(Su_index)
    
    def draw_theta(self, alpha):
        theta = np.random.dirichlet([alpha]*self.K)
        return theta
    