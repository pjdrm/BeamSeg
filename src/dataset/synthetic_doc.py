'''
Created on Jan 20, 2017

@author: root
'''
import numpy as np
from scipy import sparse, int32
from model.topic_tracking_segmentor import TopicTrackingModel
import copy


class SyntheticDocument(object):
    def __init__(self, configs):
        self.pi = configs["model"]["pi"]
        self.alpha = configs["model"]["alpha"]
        self.beta = configs["model"]["beta"]
        self.gamma = configs["model"]["gamma"]
        
        self.K = configs["synthetic_data"]["K"]
        self.W = configs["synthetic_data"]["W"]
        self.n_sents = configs["synthetic_data"]["n_sents"]
        self.sentence_l = configs["synthetic_data"]["sentence_l"]
        
        self.isMD = False
        #Just for code compatibility
        self.vocab = {}
        self.docs_index = [self.n_sents]
        self.doc_names = ["syn_doc.txt"]
        for w in range(self.W):
            self.vocab["w" + str(w)] = w
        self.inv_vocab =  {v: k for k, v in self.vocab.items()}
        self.sents_len = np.array([self.sentence_l]*self.n_sents)
        self.rho = np.random.binomial(1, self.pi, size=self.n_sents)
        #I assume that a sentence u with rho_u = 1 belong to the previous segment.
        #rho_u = 1 means a segment is coming next, this does not make sense for 
        #the last sentence. Thus, we set it to 0.
        self.rho[-1] = 0
        #need to append last sentence, otherwise last segment wont be taken into account
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        self.phi = np.array([np.random.dirichlet([self.beta]*self.W) for k in range(self.K)])
        self.n_segs = len(self.rho_eq_1)
        self.theta = np.zeros((self.n_segs, self.K))
        theta_S0 = np.random.dirichlet([self.alpha]*self.K)
        self.theta[0, :] = theta_S0
        self.U_W_counts = np.zeros((self.n_sents, self.W), dtype=int32)
        self.U_K_counts = np.zeros((self.n_sents, self.K), dtype=int32)
        self.U_I_topics = np.zeros((self.n_sents, self.sentence_l), dtype=int32)
        self.U_I_words = np.zeros((self.n_sents, self.sentence_l), dtype=int32)
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
            for word_draw in range(self.sentence_l):
                z_u_i = np.nonzero(np.random.multinomial(1, theta_Su))[0][0]
                u_topic_counts[z_u_i] += 1
                w_u_i = np.nonzero(np.random.multinomial(1, self.phi[z_u_i]))[0][0]
                u_word_count[w_u_i] += 1
                self.U_I_topics[u, word_draw] = z_u_i
                self.U_I_words[u, word_draw] = w_u_i
                self.W_K_counts[w_u_i, z_u_i] += 1
            self.U_W_counts[u, :] = u_word_count
            self.U_K_counts[u, :] = u_topic_counts
            
    def getText(self):
        str_text = "==========\n"
        for i, rho in enumerate(self.rho):
            for j in range(self.sents_len[i]):
                w_ij = self.U_I_words[i, j]
                str_text += self.inv_vocab[w_ij] + " "
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
        configs = {"model": {}, "synthetic_data": {}}
        configs["model"]["pi"] = pi
        configs["model"]["alpha"] = alpha
        configs["model"]["beta"] = beta
        configs["model"]["gamma"] = 10
        
        configs["synthetic_data"]["K"] = 10
        configs["synthetic_data"]["W"] = 200
        configs["synthetic_data"]["n_sents"] = 200
        configs["synthetic_data"]["sentence_l"] = 15
        SyntheticDocument.__init__(self, configs)
        
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
            
    def generate_docs(self, n_docs):
        self.doc_len = self.n_sents
        self.n_docs = n_docs
        self.n_sents = self.doc_len*self.n_docs
        self.doc_names = []
        for i in range(self.n_docs):
            self.doc_names.append("d" + str(i) + ".txt")
            
        if n_docs > 1:
            self.isMD = True
        else:
            self.isMD = False
            
        self.pi = np.random.beta(self.gamma, self.gamma, (n_docs,1))
        self.rho = []
        self.rho_eq_1 = []
        self.docs_index = range(self.doc_len, self.n_sents+1, self.doc_len)
        for doc, doc_index, doc_pi in zip(range(n_docs), self.docs_index, self.pi):
            doc_rho = np.random.binomial(1, doc_pi, size=self.doc_len).tolist()
            doc_rho[-1] = 0
            self.rho += doc_rho
            self.rho_eq_1 =+ np.append(np.nonzero(self.rho)[0], [doc_index])
            
            
        self.n_segs = len(self.rho_eq_1)
        self.theta = np.zeros((self.n_segs, self.K))
        theta_S0 = np.random.dirichlet([self.alpha]*self.K)
        self.theta[0, :] = theta_S0
        self.U_W_counts = np.zeros((self.n_sents, self.W), dtype=int32)
        self.U_K_counts = np.zeros((self.n_sents, self.K), dtype=int32)
        self.U_I_topics = np.zeros((self.n_sents, self.sentence_l), dtype=int32)
        self.U_I_words = np.zeros((self.n_sents, self.sentence_l), dtype=int32)
        
        initial_alpha = self.alpha
        for Su_index in range(1, self.n_segs):
            if self.rho_eq_1[Su_index] in self.docs_index:
                self.alpha = initial_alpha
                self.theta[Su_index, :] = np.random.dirichlet([self.alpha]*self.K)
            else:
                Su_begin, Su_end = self.get_Su_begin_end(Su_index)
                theta_t_minus_1 = self.theta[Su_index - 1, :]
                theta_Su = self.draw_theta(self.alpha, theta_t_minus_1)
                self.theta[Su_index, :] = theta_Su
                self.generate_Su(Su_index, theta_Su)
                self.alpha = self.update_alpha(theta_t_minus_1, self.alpha, Su_begin, Su_end)
                self.theta[Su_index, :] = self.update_theta(theta_t_minus_1, self.alpha, Su_begin, Su_end)
            
class SyntheticRndTopicPropsDoc(SyntheticDocument):
    def __init__(self, configs):
        SyntheticDocument.__init__(self, configs)

    def generate_doc(self):
        for Su_index in range(self.n_segs):
            #print("Su_index %d Su %d" % (Su_index, Su))
            theta_Su = self.draw_theta(self.alpha)
            self.theta[Su_index, :] = theta_Su
            self.generate_Su(Su_index, theta_Su)
    
    def draw_theta(self, alpha):
        theta = np.random.dirichlet([alpha]*self.K)
        return theta
'''
This class represents a synthetic collection of
related documents. The theta parameters for each segment
are the same for all documents.

In practice the multiple documents are just stored in a 
single giant matrix.
'''   
class SyntheticRndTopicMultiDoc(SyntheticDocument):
    def __init__(self, configs):
        self.doc_len = configs["synthetic_data"]["n_sents"] 
        self.n_docs = configs["synthetic_data"]["n_docs"]
        self.n_sents = self.doc_len*self.n_docs
        configs["synthetic_data"]["n_sents"] = self.n_sents
        SyntheticDocument.__init__(self, configs)
        self.doc_names = []
        for i in range(self.n_docs):
            self.doc_names.append("d" + str(i) + ".txt")
        
        #The last sentence of each document must be 1
        for u in range(self.doc_len-1, self.n_sents, self.doc_len):
            self.rho[u] = 1
        #... except last sentence.
        self.rho[-1] = 0
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        self.docs_n_segs = self.get_docs_n_segs(self.n_docs, self.doc_len)
        self.theta = self.generate_theta(max(self.docs_n_segs), self.K, self.alpha)
        self.docs_index = range(self.doc_len, self.n_sents+1, self.doc_len)
        self.isMD = True
        
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
                
class SyntheticDittoDocs(SyntheticDocument):
    def __init__(self, doc, n_copies):
        self.alpha = doc.alpha
        self.K = doc.K
        self.W = doc.W
        #Just for code compatibility
        self.vocab = doc.vocab
        self.inv_vocab =  doc.inv_vocab
        self.n_sents = doc.n_sents*n_copies
        self.sent_len = doc.sent_len
        self.sents_len = np.tile(doc.sents_len, n_copies)
        
        self.rho = np.tile(doc.rho, n_copies)
        #The last sentence of each document must be 1
        for u in range(doc.n_sents-1, doc.n_sents*n_copies, doc.n_sents):
            self.rho[u] = 1
        #... except last sentence.
        self.rho[-1] = 0
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        self.U_W_counts = np.tile(doc.U_W_counts, (n_copies, 1))
        self.U_K_counts = np.tile(doc.U_K_counts, (n_copies, 1))
        self.U_I_topics = np.tile(doc.U_I_topics, (n_copies, 1))
        self.U_I_words = np.tile(doc.U_I_words, (n_copies, 1))
        self.W_K_counts = np.tile(doc.W_K_counts, (n_copies, 1))
        self.docs_index = range(doc.n_sents, self.n_sents+1, doc.n_sents)
        self.isMD = True
                
def multi_doc_slicer(multi_doc):
    doc_l = []
    doc_begin = 0
    for doc_end, doc_name in zip(multi_doc.docs_index, multi_doc.doc_names):
        doc = copy.deepcopy(multi_doc)
        doc.n_sents = doc_end - doc_begin
        doc.n_docs = 1
        doc.sents_len = doc.sents_len[doc_begin:doc_end]
        doc.docs_index = [doc.n_sents]
        doc.doc_names = [doc_name]
        doc.rho = doc.rho[doc_begin:doc_end]
        doc.rho[-1] = 0
        doc.rho_eq_1 = np.append(np.nonzero(doc.rho)[0], [doc.n_sents-1])
        doc.U_W_counts = doc.U_W_counts[doc_begin:doc_end, :]
        doc.U_I_words = doc.U_I_words[doc_begin:doc_end, :]
        doc.isMD = False
        doc_begin = doc_end
        doc_l.append(doc)
    return doc_l