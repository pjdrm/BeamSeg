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

DP_VI_SEG = "dp_vi"
VI_SEG = "vi_seg"
QZ_LL = "segment_u_vi_qz_ll"
QZ_VOTING = "segment_u_vi_voting"
QZ_VOTING_V2 = "segment_u_vi_voting_v2"
SEG_VI = "vi_segmentation_step"

class MultiDocVISeg(AbstractSegmentor):
    
    def __init__(self, alpha,\
                       data,\
                       max_topics=None,\
                       seg_dur=10.0,\
                       std=3.0,\
                       use_prior=True,\
                       n_iters=3,\
                       seg_config=None,\
                       log_dir="../logs/",\
                       log_flag=True):
        super(MultiDocVISeg, self).__init__(alpha,\
                                            data,\
                                            max_topics=max_topics,\
                                            seg_dur=seg_dur, std=std,\
                                            use_prior=use_prior,\
                                            log_dir=log_dir,\
                                            desc="VI_seg")
        self.max_row_cache = 40
        if seg_config is None:
            self.seg_func = self.segment_u_vi_v2
            self.segmentation_step = self.dp_segmentation_step_cache
        else:
            seg_type = seg_config["type"]
            if seg_type == DP_VI_SEG:
                seg_func = seg_config["seg_func"]
                self.segmentation_step = self.dp_segmentation_step_cache
                if seg_func == QZ_LL:
                    self.seg_func = self.segment_u_vi_qz_ll
                elif seg_func == QZ_VOTING:
                    self.seg_func = self.segment_u_vi_voting
                elif seg_func == QZ_VOTING_V2:
                    self.seg_func = self.segment_u_vi_voting_v2
            elif seg_type == VI_SEG:
                self.segmentation_step = self.vi_segmentation_step
            
        self.n_iters = n_iters
        self.log_flag = log_flag
        self.vote_slack = 0.25
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
                if u_begin is None:
                    self.get_wi_segment(wi, u_clusters)
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
        doc_i_rho = [self.get_final_segmentation(doc_i) for doc_i in range(self.data.n_docs)]
        for k in range(self.max_topics):
            for doc_i in range(self.data.n_docs):
                for ui, u in enumerate(self.data.d_u_wi_indexes[doc_i]):
                    for wi in u:
                        wi_num_k = self.var_param_update(doc_i_rho[doc_i], doc_i, ui, wi, k, u_clusters)
                        norm_const[wi] += wi_num_k
                        
        for k in range(self.max_topics):
            self.qz[k] = self.qz[k]/np.array([norm_const]).T
            
    def segmentation_ll_qz(self, u_clusters):
        '''
        Returns the log likelihood of the segmentation of all documents.
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        segmentation_ll = 0.0
        for u_cluster in u_clusters:
            qz_counts = np.sum(self.qz[u_cluster.k][u_cluster.wi_list], axis=0)
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
    
    def get_k_votes_sorted(self, doc_i, u_begin, u_end):
        wi_list = self.data.d_u_wi_indexes[doc_i][u_begin:u_end+1]
        wi_list = list(chain(*wi_list))
        
        possible_clusters = range(self.max_topics)
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
            
        k_votes = sorted(k_votes.items(), key=operator.itemgetter(1), reverse=True)
        return k_votes
    
    def get_k_votes_slack(self, doc_i, u_begin, u_end, u_clusters):
        wi_list = self.data.d_u_wi_indexes[doc_i][u_begin:u_end+1]
        wi_list = list(chain(*wi_list))
        
        possible_clusters = self.get_valid_insert_clusters(doc_i, u_clusters)
        k_votes = {key: 0 for key in possible_clusters}
        for wi in wi_list:
            wi_qz = {key: 0 for key in possible_clusters}
            for k in possible_clusters:
                qz = self.qz[k][wi][self.data.W_I_words[wi]]
                wi_qz[k] = qz
            sorted_wi_qz = sorted(wi_qz.items(), key=lambda x: x[1], reverse=True)
            
            if len(possible_clusters) >= 2 and abs(sorted_wi_qz[0][1]-sorted_wi_qz[1][1]) <= self.vote_slack:
                k_votes[sorted_wi_qz[0][0]] += 0.5
                k_votes[sorted_wi_qz[1][0]] += 0.5
            else:
                k_votes[sorted_wi_qz[0][0]] += 1
                
        k_votes = sorted(k_votes.items(), key=operator.itemgetter(1), reverse=True)
        return k_votes
        
    def get_best_k_voting(self, doc_i, u_begin, u_end):
        k_votes = self.get_k_votes_sorted(doc_i, u_begin, u_end)
        return k_votes[0][0]
    
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
        
    def get_var_seg(self, u_begin, u_end, u_clusters):
        for doc_i in range(self.data.n_docs):
            doc_i_len = self.data.doc_len(doc_i)
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            
            best_k = self.get_best_k_voting(doc_i, u_begin, u_end)
            possible_clusters = self.get_valid_insert_clusters(doc_i, u_clusters)
            u_clusters = self.assign_target_k(u_begin, u_end, doc_i, best_k, possible_clusters, u_clusters)
        return u_clusters
    
    def get_var_seg_greedy(self, u_begin, u_end, prev_u_clusters):
        best_u_clusters = prev_u_clusters
        for doc_i in range(self.data.n_docs-1):
            doc_i_len = self.data.doc_len(doc_i)
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            
            best_seg_ll = -np.inf
            best_u_k_clusters = None
            #k_votes = self.get_k_votes_sorted(doc_i, u_begin, u_end)[:2]
            for k in range(self.max_topics):
                u_clusters = copy.deepcopy(best_u_clusters)
                possible_clusters = self.get_valid_insert_clusters(doc_i, u_clusters)
                u_clusters = self.assign_target_k(u_begin, u_end, doc_i, k, possible_clusters, u_clusters)
                seg_ll = self.segmentation_ll(u_clusters)
                if seg_ll > best_seg_ll:
                    best_seg_ll = seg_ll
                    best_u_k_clusters = u_clusters
            best_u_clusters = best_u_k_clusters
             
        doc_i = self.data.n_docs-1
        seg_hyps = []
        for k in range(self.max_topics):
            u_clusters = copy.deepcopy(best_u_clusters)
            possible_clusters = self.get_valid_insert_clusters(doc_i, u_clusters)
            u_clusters = self.assign_target_k(u_begin, u_end, doc_i, k, possible_clusters, u_clusters)
            seg_ll = self.segmentation_ll(u_clusters)
            seg_hyps.append((seg_ll,u_clusters))
        
        seg_hyps = sorted(seg_hyps, key=operator.itemgetter(0), reverse=True)
        return seg_hyps
    
    def get_final_segmentation(self, doc_i):
        u_clusters = self.best_segmentation[-1][0][1]
        hyp_seg = self.get_segmentation(doc_i, u_clusters)
        return hyp_seg
    
    def segment_u_vi_voting(self, u_begin, u_end, prev_u_clusters):
        '''
        The variational parameters of each in the segments (u_begin, u_end)
        vote for a topic. The segment is added to most voted topic.
        The segmentation likelihood is fully based on the real word counts.
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        u_clusters = copy.deepcopy(prev_u_clusters)
        for doc_i in range(self.data.n_docs):
            doc_i_len = self.data.doc_len(doc_i)
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            
            best_k = self.get_best_k_voting(doc_i, u_begin, u_end)
            possible_clusters = self.get_valid_insert_clusters(doc_i, u_clusters)
            u_clusters = self.assign_target_k(u_begin, u_end, doc_i, best_k, possible_clusters, u_clusters)
            
        total_ll = self.segmentation_ll(u_clusters)
        return total_ll, u_clusters
    
    def segment_u_vi_voting_v2(self, u_begin, u_end):
        '''
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        final_clusters = self.best_segmentation[u_begin-1]
        for doc_i in range(self.data.n_docs):
            doc_i_len = self.data.doc_len(doc_i)
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            best_seg_ll = -np.inf
            best_clusters = None
            k_votes_sorted = self.get_k_votes_sorted(doc_i, u_begin, u_end, final_clusters)
            possible_clusters = k_votes_sorted[0][0]
            if len(k_votes_sorted) > 1:
                possible_clusters.append(k_votes_sorted[1][0])
                
            for k in possible_clusters:
                u_clusters = copy.deepcopy(final_clusters)
                u_k_cluster = None
                for u_cluster in u_clusters:
                    if u_cluster.k == k:
                        u_k_cluster = u_cluster
                        break
                    
                if u_k_cluster is None:
                    u_k_cluster = SentenceCluster(u_begin, u_end, [doc_i], self.data, k)
                    u_clusters.append(u_k_cluster)
                else:
                    u_k_cluster.add_sents(u_begin, u_end, doc_i)
                    
                seg_ll = self.segmentation_ll(u_clusters)
                if seg_ll > best_seg_ll:
                    best_seg_ll = seg_ll
                    best_clusters = u_clusters
            final_clusters = best_clusters
            
        return best_seg_ll, best_clusters
    
    def segment_u_vi_qz_ll(self, u_begin, u_end, prev_u_clusters):
        '''
        When computing the likelihood of the segmentation,
        uses the real counts for previous segmentation (u_begin-1)
        and qz weighted counts for the segment (u_begin, u_end).
        All qz weights is distributed for all topics.
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        u_clusters_hyps = copy.deepcopy(prev_u_clusters)
        total_ll = 0.0
        for k in range(self.max_topics):
            word_counts = np.zeros(self.data.W)
            for u_cluster in u_clusters_hyps:
                if u_cluster.k == k:
                    word_counts += u_cluster.get_word_counts()
                    break
            
            for doc_i in range(self.data.n_docs):
                doc_i_len = self.data.doc_len(doc_i)
                #Accounting for documents with different lengths
                if u_begin > doc_i_len-1:
                    continue
            
                wi_list = self.data.d_u_wi_indexes[doc_i][u_begin:u_end+1]
                wi_list = list(chain(*wi_list))
                word_counts += np.sum(self.qz[k][wi_list], axis=0)
            total_ll += self.segment_ll(word_counts)
            
        u_clusters_hyps = self.get_var_seg_greedy(u_begin, u_end, u_clusters_hyps)
        if self.use_prior:
            total_ll += self.segmentation_log_prior(u_clusters_hyps[0][1])
        return total_ll, u_clusters_hyps
    
    def vi_segmentation_step(self):
        '''
        Looks at each sentence individually and adds it to the
        u_cluster with the highest segmentation log likelihood.
        In each step only two u_clusters are considered based
        the highest values of qz.
        '''
        final_u_clusters = []
        for doc_i in range(self.data.n_docs):
            for u in range(self.data.doc_len(doc_i)):
                k_votes_sorted = self.get_k_votes_sorted(doc_i, u, u, final_u_clusters)
                best_ks = [k_votes_sorted[0][0]]
                if len(k_votes_sorted) > 1:
                    best_ks.append(k_votes_sorted[1][0])
                
                best_seg_ll = -np.inf
                best_u_clusters = None
                for k in best_ks:
                    current_u_clusters = copy.deepcopy(final_u_clusters)
                    u_k_cluster = None
                    for u_cluster in current_u_clusters:
                        if u_cluster.k == k:
                            u_k_cluster = u_cluster
                            break
                        
                    if u_k_cluster is None:
                        u_k_cluster = SentenceCluster(u, u, [doc_i], self.data, k)
                        current_u_clusters.append(u_k_cluster)
                    else:
                        u_k_cluster.add_sents(u, u, doc_i)
                        
                    seg_ll = self.segmentation_ll(current_u_clusters)
                    if seg_ll > best_seg_ll:
                        best_seg_ll = seg_ll
                        best_u_clusters = current_u_clusters
                final_u_clusters = best_u_clusters
        self.best_segmentation[-1] = final_u_clusters
        
    def segment_last_line(self, u_begin, u_end, best_seg):
        best_u_clusters =  None
        best_seg_ll = -np.inf
        for prev_seg_ll, prev_u_clusters in best_seg:
            u_clusters = copy.deepcopy(prev_u_clusters)
            seg_ll, u_clusters = self.get_var_seg_greedy(u_begin, u_end, u_clusters)
            if self.use_prior:
                seg_ll += self.segmentation_log_prior(u_clusters)
            if seg_ll > best_seg_ll:
                best_u_clusters = u_clusters
                best_seg_ll = seg_ll
        return [(best_seg_ll, best_u_clusters)]
        
    def check_if_lost_gs_seg(self, u, cached_segs):
        found_correct_seg = False
        for seg_ll, seg_clusters in cached_segs:
            n_correct_segs = 0
            for doc_j in range(0, self.data.n_docs):
                rho_seg = self.get_segmentation(doc_j, seg_clusters)
                rho_gs = list(self.data.docs_rho_gs[doc_j][:u+1])
                rho_seg[-1] = 1
                rho_gs[-1] = 1
                if str(rho_seg) == str(rho_gs):
                    n_correct_segs += 1
            if n_correct_segs == self.data.n_docs: 
                found_correct_seg = True
                break
        if not found_correct_seg:
            print("\nLOST CORRECT SEG u: %d"%u)
            
    def dp_segmentation_step_cache(self):
        with open(self.log_dir+"dp_tracker_"+self.desc+".txt", "a+") as f:
            f.write("DP tracking:\n")
            for u_end in range(self.data.max_doc_len-1):
                f.write("Tracking line %d\n"%(u_end))
                cached_segs = []
                best_seg_ll = -np.inf
                for u_begin in range(u_end+1):
                    if u_end == 3 and u_begin == 0:
                        a = 0
                        
                    if u_begin == 0:
                        best_seg = [(-np.inf, [])]
                    else:
                        best_seg = self.best_segmentation[u_begin-1]
                        
                    best_mat_entry_seg_ll = -np.inf
                    u_clusters_mat_entry = []
                    for prev_seg_ll, prev_u_cluster in best_seg:
                        seg_ll, seg_hyps = self.seg_func(u_begin, u_end, prev_u_cluster)
                        
                        for seg_hyp in seg_hyps:
                            if not self.is_cached_seg(seg_hyp[0], u_clusters_mat_entry):
                                u_clusters_mat_entry.append(seg_hyp)
                                u_clusters_mat_entry = sorted(u_clusters_mat_entry, key=operator.itemgetter(0), reverse=True)
                                
                        if seg_ll > best_mat_entry_seg_ll:
                            best_mat_entry_seg_ll = seg_ll
                            
                    '''
                    if best_mat_entry_seg_ll > best_seg_ll:
                        best_seg_ll = best_mat_entry_seg_ll
                        cached_segs = u_clusters_mat_entry[:self.max_row_cache]
                    '''
                    cached_segs += u_clusters_mat_entry
                            
                    f.write("(%d,%d)\tll: %.3f\n"%(u_begin, u_end, best_mat_entry_seg_ll))
                    for doc_i in range(self.data.n_docs):
                        f.write(str(self.get_segmentation(doc_i, u_clusters_mat_entry[0][1]))+" "
                                +str(self.get_seg_with_topics(doc_i, u_clusters_mat_entry[0][1]))+"\n")
                    f.write("\n")
                #if u_end > 1:
                #    self.check_if_lost_gs_seg(u_end, cached_segs)
                cached_segs = sorted(cached_segs, key=operator.itemgetter(0), reverse=True)
                cached_segs = cached_segs[:self.max_row_cache]
                f.write("---cache---\n")
                for cached_seg in cached_segs:
                    f.write("ll: %.3f\n"%(cached_seg[0]))
                    for doc_i in range(self.data.n_docs):
                        f.write(str(self.get_segmentation(doc_i, cached_seg[1]))+" "
                                +str(self.get_seg_with_topics(doc_i, cached_seg[1]))+"\n")
                    f.write("\n")
                f.write("-----------\n")
                f.write("============\n")
                self.best_segmentation[u_end] = cached_segs
                
            #Code to segment the last line
            u_end = self.data.max_doc_len-1
            best_seg_ll = -np.inf
            best_u_clusters = None
            for u_begin in range(u_end+1):
                best_mat_entry_seg_ll = -np.inf
                u_clusters_mat_entry = None
                if u_begin == 0:
                    best_seg = [(-np.inf, [])]
                else:
                    best_seg = self.best_segmentation[u_begin-1]
                    
                for prev_seg_ll, prev_u_clusters in best_seg:
                    u_clusters = copy.deepcopy(prev_u_clusters)
                    seg_hyps = self.get_var_seg_greedy(u_begin, u_end, u_clusters)
                    seg_ll = seg_hyps[0][0]
                    u_clusters = seg_hyps[0][1]
                    
                    if self.use_prior:
                        seg_ll += self.segmentation_log_prior(u_clusters)
                        
                    if seg_ll > best_seg_ll:
                        best_u_clusters = u_clusters
                        best_seg_ll = seg_ll
                        
                    if seg_ll > best_mat_entry_seg_ll:
                        u_clusters_mat_entry = u_clusters
                        best_mat_entry_seg_ll = seg_ll
                
                f.write("(%d,%d)\tll: %.3f\n"%(u_begin, u_end, best_mat_entry_seg_ll))
                for doc_i in range(self.data.n_docs):
                    f.write(str(self.get_segmentation(doc_i, u_clusters_mat_entry))+" "
                            +str(self.get_seg_with_topics(doc_i, u_clusters_mat_entry))+"\n")
                f.write("\n")
                
            self.check_if_lost_gs_seg(u_end, cached_segs)
            f.write("============\n")
            self.best_segmentation[u_end] = [(best_seg_ll, best_u_clusters)]
            
        
    def segment_docs(self): #TODO: check if the segmentation changes and use that as criteria for stopping
        '''
        Segments the full collection of documents. Alternates between using a Dynamic Programming
        procedure for segmentation and performing variational inference to update the
        certainty about words belonging to a topic (segments/language model).
        :param n_iters: number of iterations to perform
        '''
        self.set_gl_data(self.data)
        segmentation_log_files = [log_tools.log_init(self.log_dir+"segs_doc"+str(doc_i)+".txt") for doc_i in range(self.data.n_docs)]
        vi_log_files = []
        for doc_i in range(self.data.n_docs):
            doc_i_topics = []
            for k in range(self.max_topics):
                doc_i_topics.append(log_tools.log_init(self.log_dir+"vi_doc_"+str(doc_i)+"_k"+str(k)+".txt"))
            vi_log_files.append(doc_i_topics)
            
        t = trange(self.n_iters, desc='', leave=True)
        for i in t:
            if i == 2:
                a = 0
            start = time.time()
            self.segmentation_step()
            if self.log_flag:
                for doc_i in range(self.data.n_docs):
                    doc_i_log = segmentation_log_files[doc_i]
                    seg_log_str = "\nGS: "+str(self.data.docs_rho_gs[doc_i].tolist())+\
                                  "\nVI: "+str(self.get_segmentation(doc_i, self.best_segmentation[-1][0][1]))+\
                                  "\n K: "+self.get_seg_with_topics(doc_i, self.best_segmentation[-1][0][1])
                    doc_i_log.info(seg_log_str)
                    wi_list = self.data.d_u_wi_indexes[doc_i]
                    wi_list = list(chain(*wi_list))
                    for k in range(self.max_topics):
                        vi_log_str = "\n"+str(self.qz[k][wi_list])
                        vi_log_files[doc_i][k].info(vi_log_str)
            end = time.time()
            dp_step_time = (end - start)
            
            best_segmentation = self.best_segmentation[-1][0][1]
            start = time.time()
            self.variational_step(best_segmentation)
            end = time.time()
            variational_step_time = (end - start)
            t.set_description("DP_time: %.1f VI_time: %.1f" % (dp_step_time, variational_step_time))
            if i < self.n_iters-1:
                self.best_segmentation = [[] for i in range(self.data.max_doc_len)]
        print("\n")