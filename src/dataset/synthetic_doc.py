'''
Created on Jan 20, 2017

@author: root
'''
import numpy as np
from scipy import sparse, int32
from model.topic_tracking_segmentor import TopicTrackingModel


class SyntheticDocument(object):
    def __init__(self, pi, alpha, beta, K, W, n_sents, sent_len):
        self.alpha = alpha
        self.K = K
        self.W = W
        #Just for code compatibility
        self.vocab = {}
        for w in range(self.W):
            self.vocab[str(w)] = w
        self.inv_vocab =  {v: k for k, v in self.vocab.items()}
        self.n_sents = n_sents
        self.sent_len = sent_len
        self.sents_len = np.array([sent_len]*n_sents)
        self.rho = np.random.binomial(1, pi, size=n_sents)
        #I assume that a sentence u with rho_u = 1 belong to the previous segment.
        #rho_u = 1 means a segment is coming next, this does not make sense for 
        #the last sentence. Thus, we set it to 0.
        self.rho[-1] = 0
        #need to append last sentence, otherwise last segment wont be taken into account
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [n_sents-1])
        self.phi = np.array([np.random.dirichlet([beta]*W) for k in range(K)])
        self.n_segs = len(self.rho_eq_1)
        self.theta = np.zeros((self.n_segs, self.K))
        theta_S0 = np.random.dirichlet([self.alpha]*self.K)
        self.theta[0, :] = theta_S0
        self.U_W_counts = np.zeros((self.n_sents, self.W), dtype=int32)
        self.U_K_counts = np.zeros((self.n_sents, self.K), dtype=int32)
        self.U_I_topics = np.zeros((self.n_sents, self.sent_len), dtype=int32)
        self.U_I_words = np.zeros((self.n_sents, self.sent_len), dtype=int32)
        #Matrix with the number of times each word in the vocab was assigned with topic k
        self.W_K_counts = np.zeros((self.W, self.K), dtype=int32)
    
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
    def generate_Su(self, Su_index, theta_Su):
        Su_begin, Su_end = self.get_Su_begin_end(Su_index)
        print("Generating words for %d - %d segment" % (Su_begin, Su_end))
        for u in range(Su_begin, Su_end):
            u_word_count = np.zeros(self.W)
            u_topic_counts = np.zeros(self.K)
            for word_draw in range(self.sent_len):
                z_u_i = np.nonzero(np.random.multinomial(1, theta_Su))[0][0]
                u_topic_counts[z_u_i] += 1
                w_u_i = np.nonzero(np.random.multinomial(1, self.phi[z_u_i]))[0][0]
                u_word_count[w_u_i] += 1
                self.U_I_topics[u, word_draw] = z_u_i
                self.U_I_words[u, word_draw] = w_u_i
                self.W_K_counts[w_u_i, z_u_i] += 1
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
        self.generate_Su(Su_index, theta_S0)
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
            theta_Su = self.draw_theta(self.alpha, theta_t_minus_1)
            self.theta[Su_index, :] = theta_Su
            self.generate_Su(Su_index, theta_Su)
            self.alpha = self.update_alpha(theta_t_minus_1, self.alpha, Su_begin, Su_end)
            self.theta[Su_index, :] = self.update_theta(theta_t_minus_1, self.alpha, Su_begin, Su_end)
            
class SyntheticRndTopicPropsDoc(SyntheticDocument):
    def __init__(self, pi, alpha, beta, K, W, n_sents, sentence_l):
        SyntheticDocument.__init__(self, pi, alpha, beta, K, W, n_sents, sentence_l)

    def generate_doc(self):
        for Su_index in range(self.n_segs):
            #print("Su_index %d Su %d" % (Su_index, Su))
            theta_Su = self.draw_theta(self.alpha)
            self.theta[Su_index, :] = theta_Su
            self.generate_Su(Su_index, theta_Su)
    
    def draw_theta(self, alpha):
        theta = np.random.dirichlet([alpha]*self.K)
        return theta
    
class SyntheticRndTopicDocCollection(SyntheticDocument):
    def __init__(self, pi, alpha, beta, K, W, doc_len, sent_len, n_docs):
        self.alpha = alpha
        self.K = K
        self.W = W
        self.vocab = {}
        for w in range(self.W):
            self.vocab[str(w)] = w
        self.inv_vocab =  {v: k for k, v in self.vocab.items()}
        self.n_docs = n_docs
        self.n_sents = doc_len*n_docs
        self.sent_len = sent_len
        self.sents_len = np.array([sent_len]*self.n_sents)
        
        self.rho = np.random.binomial(1, pi, size=(self.n_sents))
        #The last sentence of each document must be 1
        for u in range(doc_len-1, self.n_sents, doc_len):
            self.rho[u] = 1
        #... except last sentence.
        self.rho[-1] = 0
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        self.phi = np.array([np.random.dirichlet([beta]*W) for k in range(K)])
        self.docs_n_segs = self.get_docs_n_segs(n_docs, doc_len)
        self.theta = self.generate_theta(max(self.docs_n_segs), K, alpha)
        
        self.U_W_counts = np.zeros((self.n_sents, self.W), dtype=int32)
        self.U_K_counts = np.zeros((self.n_sents, self.K), dtype=int32)
        self.U_I_topics = np.zeros((self.n_sents, self.sent_len), dtype=int32)
        self.U_I_words = np.zeros((self.n_sents, self.sent_len), dtype=int32)
        #Matrix with the number of times each word in the vocab was assigned with topic k
        self.W_K_counts = np.zeros((self.W, self.K), dtype=int32)
        
    def get_docs_n_segs(self, n_docs, doc_len):
        docs_n_segs = []
        for d in range(n_docs):
            n_segs = 0
            for u in range(doc_len):
                rho = self.rho[u+d*doc_len]
                if rho == 1:
                    n_segs += 1
            docs_n_segs.append(n_segs)
        #Compensating for the last sentece having rho = 0
        docs_n_segs[-1] += 1
        return docs_n_segs
    
    def generate_theta(self, n_segs, K, alpha):
        theta = np.zeros((n_segs, K))
        for Su_index in range(n_segs):
            theta[Su_index, :] = np.random.dirichlet([alpha]*K)
        return theta
    
    def generate_docs(self):
        Su_index = 0
        for d in range(self.n_docs):
            theta_i = 0
            for seg in range(self.docs_n_segs[d]):
                theta_Su = self.theta[theta_i, :]
                self.generate_Su(Su_index, theta_Su)
                theta_i += 1
                Su_index += 1