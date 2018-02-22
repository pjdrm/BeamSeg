'''
Created on Feb 22, 2018

@author: pjdrm
'''
from model.dp.segmentor import AbstractSegmentor, SentenceCluster
import numpy as np
from itertools import chain
import operator
import copy
from debug import log_tools
from tqdm import trange
import time

np.set_printoptions(threshold=np.inf, linewidth=200)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

class MultiDocVISeg(AbstractSegmentor):
    
    def __init__(self, beta, data, max_topics=None, n_iters=3, log_dir="../logs/"):
        super(MultiDocVISeg, self).__init__(beta, data)
        self.max_topics = self.data.max_doc_len if max_topics is None else max_topics
        self.seg_func = self.segment_u_vi_v3
        self.n_iters = n_iters
        self.log_dir = log_dir
        #List of matrices (one for each topic). Lines are words in the document
        #collection and columns the vocabulary indexes.
        #The entries contains the value of the corresponding variational parameter.
        self.qz = self.init_variational_params(self.data.total_words,\
                                               self.data.W,\
                                               self.max_topics,\
                                               self.data.W_I_words)
        
    def init_variational_params(self, total_words, W, K, W_I_words):
        qz = []
        for k in range(K):
            qz_k = np.zeros((total_words, W))
            qz.append(qz_k)
            
        for wi in range(total_words):
            qz_K = np.random.dirichlet([1.0/K]*K)
            w = W_I_words[wi]
            for k in range(K):
                qz[k][wi,w] = qz_K[k]
                
        return qz
    
    def qz_words_minus_wi_gibbs(self, rho, doc_i, u, wi, k, u_clusters):
        '''
        Returns the set of words that influence the qz update
        of wi.
        '''
        wi_u_cluster = self.get_wi_segment(wi, u_clusters)
        if k == wi_u_cluster.k:
            words = list(set(wi_u_cluster.get_words())-set([wi]))
            return words
        else:
            if rho[u] == 0:
                u_begin, u_end = wi_u_cluster.get_segment(doc_i)
                words = []
                for u in range(u_begin, u_end+1):
                    u_wi_indexes = self.data.d_u_wi_indexes[doc_i][u]
                    if wi in u_wi_indexes:
                        u_wi_indexes = list(set(u_wi_indexes)-set([wi]))
                        words += u_wi_indexes
                        break
                    words += u_wi_indexes
            else:
                words = list(set(wi_u_cluster.get_words())-set([wi]))
                next_u_cluster = self.get_next_cluster(wi_u_cluster.k, doc_i, u_clusters)
                if next_u_cluster is not None:
                    #If its the same we are just going to add it in the for loop below
                    if next_u_cluster.k != k:
                        words += next_u_cluster.get_doc_words(doc_i)
                    
            for u_cluster in u_clusters:
                if u_cluster.k == k:
                    words += u_cluster.get_words_minus_doc(doc_i) #TODO: might be worth to try just adding all words
                    break
            return words
        
    def qz_words_minus_wi_seg_flip(self, doc_i, u, wi, k, u_clusters):
        '''
        Returns the set of words that influence the qz update
        of wi.
        '''
        wi_u_cluster = self.get_wi_segment(wi, u_clusters)
        if k == wi_u_cluster.k:
            words = list(set(wi_u_cluster.get_words())-set([wi]))
            return words
        else:
            u_begin, u_end = wi_u_cluster.get_segment(doc_i)
            words = []
            for u in range(u_begin, u_end+1):
                u_wi_indexes = self.data.d_u_wi_indexes[doc_i][u]
                words += u_wi_indexes
            words = list(set(words)-set([wi]))
        
            #If wi is a different topic than the one we are updating (k)
            #we need find the words from other documents in that topic. 
            for u_cluster in u_clusters:  #TODO: seems like I am adding words from doc_i from other clusters. I dont think I want that.
                if u_cluster.k == k:
                    words += u_cluster.get_words_minus_doc(doc_i)
                    break
                
            return words
            
    def var_update_k_val(self, rho, doc_i, u, wi, k, u_clusters):
        '''
        Return the value of the numerator for the variational update expression.
        :param doc_i: document index
        :param wi: word index (relative to the full collection of documents)
        :param k: topic/language model/segment index
        :param u_clusters: list of sentence clusters representing a segmentation of all documents
        '''
        words_update = self.qz_words_minus_wi_gibbs(rho, doc_i, u, wi, k, u_clusters)
        #words_update = self.qz_words_minus_wi_seg_flip(doc_i, u, wi, k, u_clusters)
        E_counts_f2 = self.qz[k][words_update]
        Var_counts_f2 = E_counts_f2*(1.0-E_counts_f2)
        C_beta_E_counts_f2_sum = self.C_beta+np.sum(E_counts_f2)
        #E_q_f2 = np.log(C_beta_E_counts_f2_sum)-(np.sum(Var_counts_f2)/(2.0*(C_beta_E_counts_f2_sum)**2))
        
        word_mask = np.array([(self.data.W_I_words[words_update]==self.data.W_I_words[wi]).astype(np.int)]).T
        E_counts_f1 = E_counts_f2*word_mask
        Var_counts_f1 = np.sum(Var_counts_f2*word_mask)
        C_beta_E_counts_f1_sum = self.beta[self.data.W_I_words[wi]]+np.sum(E_counts_f1)
        Var_counts_f1 = Var_counts_f1/(2.0*(C_beta_E_counts_f1_sum)**2)
        Var_counts_f2 = np.sum(Var_counts_f2)/(2.0*(C_beta_E_counts_f2_sum)**2)
        #E_q_f1 = np.log(C_beta_E_counts_f1_sum)-(np.sum(Var_counts_f1)/(2.0*(C_beta_E_counts_f1_sum)**2))
        #num = np.exp(E_q_f1-E_q_f2)
        num = C_beta_E_counts_f1_sum*(1.0/C_beta_E_counts_f2_sum)*np.exp(-Var_counts_f1+Var_counts_f2)
        
        return num
        
    def var_param_update(self, rho, doc_i, u, wi, k, u_clusters):
        '''
        Updates the variational parameter of word wi for topic k.
        :param doc_i: document index
        :param wi: word index (relative to the full collection of documents)
        :param k: topic/language model/segment index
        :param u_clusters: list of sentence clusters representing a segmentation of all documents
        '''
        num_k = self.var_update_k_val(rho, doc_i, u, wi, k, u_clusters)
        self.qz[k][wi][self.data.W_I_words[wi]] = num_k
        return num_k
    
    def variational_step(self, u_clusters):
        '''
        Update the variational parameters for all words and all topics.
        :param u_clusters: list of sentence clusters representing the current segmentation state 
        '''
        norm_const = np.zeros(self.data.total_words)
        doc_i_rho = [self.get_segmentation(doc_i, self.best_segmentation[-1]) for doc_i in range(self.data.n_docs)]
        for k in range(self.max_topics):
            for doc_i in range(self.data.n_docs):
                for ui, u in enumerate(self.data.d_u_wi_indexes[doc_i]):
                    for wi in u:
                        wi_num_k = self.var_param_update(doc_i_rho[doc_i], doc_i, ui, wi, k, u_clusters)
                        norm_const[wi] += wi_num_k
                        
        for k in range(self.max_topics):
            self.qz[k] = self.qz[k]/np.array([norm_const]).T
            
    def segmentation_ll(self, u_clusters):
        '''
        Returns the log likelihood of the segmentation of all documents.
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        segmentation_ll = 0.0
        for u_cluster in u_clusters:
            #qz_counts = np.sum(self.qz[u_cluster.k][u_cluster.wi_list], axis=0)
            qz_counts = u_cluster.get_word_counts() #Version that does not use qz weight at all
            segmentation_ll += self.segment_ll(qz_counts)
        return segmentation_ll
    
    def get_best_k(self, doc_i, u_begin, u_end, u_clusters):
        wi_list = self.data.d_u_wi_indexes[doc_i][u_begin:u_end+1]
        wi_list = list(chain(*wi_list))
        best_qz_sum = 0.0
        best_k = -1
        
        possible_clusters = self.get_valid_insert_clusters(doc_i, u_clusters)
        for k in possible_clusters:
            qz_sum = np.sum(self.qz[k][wi_list])
            if qz_sum > best_qz_sum:
                best_qz_sum = qz_sum
                best_k = k
                
        if best_k == -1:
            possible_clusters = self.get_valid_insert_clusters(doc_i, u_clusters)
            print()
            
        return best_k
    
    def get_best_k_voting(self, doc_i, u_begin, u_end, u_clusters):
        wi_list = self.data.d_u_wi_indexes[doc_i][u_begin:u_end+1]
        wi_list = list(chain(*wi_list))
        
        possible_clusters = self.get_valid_insert_clusters(doc_i, u_clusters)
        k_votes = {key: 0 for key in possible_clusters}
        for wi in wi_list:
            best_cluster = -1
            best_qz = -1
            for k in possible_clusters:
                qz = self.qz[k][wi][self.data.W_I_words[wi]]
                if qz > best_qz:
                    best_cluster = k
                    best_qz = qz
            k_votes[best_cluster] += 1
        
        best_k = max(k_votes.iteritems(), key=operator.itemgetter(1))[0]
                
        if best_k == -1:
            print("WARNING: inavlid best_k")
            
        return best_k
    
    def get_best_k_voting_u0(self, u_end):
        k_votes = {key: 0 for key in range(self.max_topics)}
        for doc_i in range(self.data.n_docs):
            wi_list = self.data.d_u_wi_indexes[doc_i][0:u_end+1]
            wi_list = list(chain(*wi_list))
            for wi in wi_list:
                best_cluster = -1
                best_qz = -1
                for k in range(self.max_topics):
                    qz = self.qz[k][wi][self.data.W_I_words[wi]]
                    if qz > best_qz:
                        best_cluster = k
                        best_qz = qz
                k_votes[best_cluster] += 1
        
        best_k = max(k_votes.iteritems(), key=operator.itemgetter(1))[0]
        return best_k
        
    def get_var_seg(self, u_begin, u_end, best_seg):
        for doc_i in range(self.data.n_docs):
            doc_i_len = self.data.doc_len(doc_i)
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            
            best_k = self.get_best_k(doc_i, u_begin, u_end, best_seg)
            u_k_cluster = None
            for u_cluster in best_seg:
                if u_cluster.k == best_k:
                    u_k_cluster = u_cluster
                    break
                
            if u_k_cluster is None:
                    u_k_cluster = SentenceCluster(u_begin, u_end, [doc_i], self.data, best_k)
                    best_seg.append(u_k_cluster)
            else:
                u_k_cluster.add_sents(u_begin, u_end, doc_i)
        return best_seg
    
    def segment_u_vi_v2(self, u_begin, u_end):
        '''
        Estimates, for all documents, the best segmentation
        from u_end to u_begin (column index in the DP matrix).
        Implements a faster version that does not explore all self.doc_combs_list.
        The heuristic considers adding (or not) the current segmentation point of an
        individual documents. The corresponding sentences are added to the option
        that yields the highest likelihood. In each step we go to the next document
        and we stop until all of them are covered.
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        if u_begin == 0: #The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            u_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data, 0)
            total_ll = 0.0
            for k in range(self.max_topics):
                k_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data, k)
                total_ll += self.segmentation_ll([k_cluster])
            return total_ll, [u_cluster] #TODO: maybe I think break based on qz
      
      
        best_seg = copy.deepcopy(self.best_segmentation[u_begin-1])
        total_ll = 0.0
        for k in range(self.max_topics):
            u_k_cluster = None
            for u_cluster in best_seg:
                if u_cluster.k == k:
                    u_k_cluster = copy.deepcopy(u_cluster)
                    break
                
            for doc_i in range(self.data.n_docs):
                doc_i_len = self.data.doc_len(doc_i)
                #Accounting for documents with different lengths
                if u_begin > doc_i_len-1:
                    continue
                
                if u_k_cluster is None:
                    u_k_cluster = SentenceCluster(u_begin, u_end, [doc_i], self.data, k)
                else:
                    u_k_cluster.add_sents(u_begin, u_end, doc_i)
                
            total_ll += self.segmentation_ll([u_k_cluster])
        
        best_seg = self.get_var_seg(u_begin, u_end, best_seg)
        return total_ll, best_seg
    
    def segment_u_vi_v3(self, u_begin, u_end):
        '''
        
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        #Seems that doing this for u_begin==0 is bad. Imagine, because of initialization, we end up with a single segment on k=0.
        #The VI updates for k=0 will have all words from other segments, which will lower the confidence in k=0. Other topics will
        #have better updates since we flip the rho value and discard many words that are bad. Thus, the weights will prefer other 
        #topics (different from k=0). In the code bellow I was always forcing k=0, probably thats why its not working.
        '''
        if u_begin == 0: #The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            u_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data, 0)
            total_ll = 0.0
            k0_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data, 0)
            total_ll += self.segmentation_ll([k0_cluster])
            return total_ll, [u_cluster] #TODO: maybe I think break based on qz
        '''
        
        if u_begin == 0:
            best_k = self.get_best_k_voting_u0(u_end)
            u_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data, best_k)
            total_ll = self.segmentation_ll([u_cluster])
            return total_ll, [u_cluster]
            
        best_seg = copy.deepcopy(self.best_segmentation[u_begin-1])
        for doc_i in range(self.data.n_docs):
            doc_i_len = self.data.doc_len(doc_i)
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            
            best_k = self.get_best_k_voting(doc_i, u_begin, u_end, best_seg)
            u_k_cluster = None
            for u_cluster in best_seg:
                if u_cluster.k == best_k:
                    u_k_cluster = u_cluster
                    break
                
            if u_k_cluster is None:
                u_k_cluster = SentenceCluster(u_begin, u_end, [doc_i], self.data, best_k)
                best_seg.append(u_k_cluster)
            else:
                u_k_cluster.add_sents(u_begin, u_end, doc_i)
            
        total_ll = self.segmentation_ll(best_seg)
        
        #best_seg = self.get_var_seg(u_begin, u_end, best_seg)
        return total_ll, best_seg
    
    def segment_docs(self): #TODO: check if the segmentation changes and use that as criteria for stopping
        '''
        Segments the full collection of documents. Alternates between using a Dynamic Programming
        procedure for segmentation and performing variational inference to update the
        certainty about words belonging to a topic (segments/language model).
        :param n_iters: number of iterations to perform
        '''
        segmentation_log_files = [log_tools.log_init(self.log_dir+"segs_doc"+str(doc_i)+".txt") for doc_i in range(self.data.n_docs)]
        vi_log_files = []
        for doc_i in range(self.data.n_docs):
            doc_i_topics = []
            for k in range(self.max_topics):
                doc_i_topics.append(log_tools.log_init(self.log_dir+"vi_doc_"+str(doc_i)+"_k"+str(k)+".txt"))
            vi_log_files.append(doc_i_topics)
            
        t = trange(self.n_iters, desc='', leave=True)
        for i in t:
            start = time.time()
            self.dp_segmentation_step()
            for doc_i in range(self.data.n_docs):
                doc_i_log = segmentation_log_files[doc_i]
                seg_log_str = "\nGS: "+str(self.data.docs_rho_gs[doc_i].tolist())+\
                              "\nVI: "+str(self.get_segmentation(doc_i, self.best_segmentation[-1]))
                doc_i_log.info(seg_log_str)
                wi_list = self.data.d_u_wi_indexes[doc_i]
                wi_list = list(chain(*wi_list))
                for k in range(self.max_topics):
                    vi_log_str = "\n"+str(self.qz[k][wi_list])
                    vi_log_files[doc_i][k].info(vi_log_str)
            end = time.time()
            dp_step_time = (end - start)
            
            best_segmentation = self.best_segmentation[-1]
            start = time.time()
            self.variational_step(best_segmentation)
            end = time.time()
            variational_step_time = (end - start)
            t.set_description("DP_time: %.1f VI_time: %.1f" % (dp_step_time, variational_step_time))
            if i < self.n_iters-1:
                self.best_segmentation = [[] for i in range(self.data.max_doc_len)]
        print("\n")