'''
Created on Feb 22, 2018

@author: pjdrm
'''
import numpy as np
import copy
from scipy.special import gammaln
import os
import collections
from scipy.stats import norm

GL_DATA = None

class AbstractSegmentor(object):

    def __init__(self, beta,\
                       data,\
                       max_topics=None,\
                       seg_dur=10.0,\
                       std=3.0,\
                       use_prior=True,\
                       log_dir="../logs/",\
                       desc="Abs_seg"):
        self.set_gl_data(data)
        self.data = data
        self.seg_dur = seg_dur
        self.std = std
        self.use_prior = use_prior
        self.max_topics = self.data.max_doc_len if max_topics is None else max_topics
        self.desc = desc
        self.log_dir = log_dir
        self.beta = beta
        self.C_beta = np.sum(self.beta)
        self.W = data.W
        self.best_segmentation = [[] for i in range(self.data.max_doc_len)]
        self.seg_ll_C = gammaln(self.beta.sum())-gammaln(self.beta).sum()
        
        os.remove(self.log_dir+"dp_tracker_"+self.desc+".txt") if os.path.exists(self.log_dir+"dp_tracker_"+self.desc+".txt") else None

    def set_gl_data(self, data):
        global GL_DATA
        GL_DATA = data
        
    def print_seg(self, u_clusters):
        print("==========================")
        for doc_i in range(self.data.n_docs):
            seg = self.get_final_segmentation(doc_i, u_clusters)
            print("Doc %d: %s" % (doc_i, str(seg)))
            
    def get_seg_with_topics(self, doc_i, u_clusters):
        if isinstance(u_clusters[0], collections.Iterable):
            u_clusters = u_clusters[0]
            
        rho = []
        segments_dict = {}
        for u_cluster in u_clusters:
            if u_cluster.has_doc(doc_i):
                segments_dict[u_cluster.k] = u_cluster.get_segment(doc_i)
        sorted_segments_dict = sorted(segments_dict.items(), key=lambda x: x[1][0])
        for seg in sorted_segments_dict:
            u_begin = seg[1][0]
            u_end = seg[1][1]
            k = seg[0]
            for u in range(u_begin, u_end+1):
                rho.append(k)
        return rho
            
    def get_cluster_order(self, doc_i, u_clusters):
        cluster_k_list = []
        for u_cluster in u_clusters:
            if u_cluster.has_doc(doc_i):
                u_begin, u_end = u_cluster.get_segment(doc_i)
                cluster_k_list.append([u_cluster.k, u_begin])
        cluster_k_list = sorted(cluster_k_list, key=lambda x: x[1])
        ret_list = []
        for cluster_k in cluster_k_list:
            ret_list.append(cluster_k[0])
        return ret_list
    
    def get_final_segmentation(self, doc_i):
        '''
        Returns the final segmentation for a document.
        This is done by backtracking the best segmentations
        in a bottom-up fashion.
        :param doc_i: document index
        '''
        u_clusters = self.best_segmentation[-1]
        hyp_seg = self.get_segmentation(doc_i, u_clusters)
        return hyp_seg
    
    def get_segmentation(self, doc_i, u_clusters):
        '''
        Returns the final segmentation for a document.
        This is done by backtracking the best segmentations
        in a bottom-up fashion.
        :param doc_i: document index
        '''
        if isinstance(u_clusters[0], collections.Iterable):
            u_clusters = u_clusters[0]
            
        hyp_seg = []
        cluster_order = self.get_cluster_order(doc_i, u_clusters)
        for k in cluster_order:
            u_cluster = self.get_k_cluster(k, u_clusters)
            u_begin, u_end = u_cluster.get_segment(doc_i)
            seg_len = u_end-u_begin+1
            doc_i_seg = list([0]*seg_len)
            doc_i_seg[-1] = 1
            hyp_seg += doc_i_seg
        return hyp_seg
    
    def get_all_segmentations(self):
        '''
        Returns a single vector with the final
        segmentation for all documents.
        '''
        all_segs = []
        for doc_i in range(self.data.n_docs):
            all_segs += self.get_final_segmentation(doc_i)
        return all_segs
            
    def get_last_cluster(self, doc_i, u_clusters):
        '''
        Returns the last cluster index where doc_i is present
        :param doc_i: document index
        :param u_clusters: list of sentence cluster corresponding to a segmentation
        '''
        last_sent = -1
        last_cluster = -1
        for cluster_i, u_cluster in enumerate(u_clusters):
            if u_cluster.has_doc(doc_i):
                u_begin, u_end = u_cluster.get_segment(doc_i)
                if u_end > last_sent:
                    last_cluster = cluster_i
                    last_sent = u_end
        if last_cluster == -1:
            last_cluster = 0
        return last_cluster
    
    def get_valid_insert_clusters(self, doc_i, u_clusters):
        if len(u_clusters) == 0:
            return range(0, self.max_topics)
        last_sent = -1
        last_cluster_i = -1
        u_clusters_with_doc_i = []
        for u_cluster in u_clusters:
            if u_cluster.has_doc(doc_i):
                u_clusters_with_doc_i.append(u_cluster.k)
                u_begin, u_end = u_cluster.get_segment(doc_i)
                if u_end > last_sent:
                    last_cluster_i = u_cluster.k
                    last_sent = u_end
                    
        invalid_clusters = set(u_clusters_with_doc_i)
        if last_cluster_i != -1:
            invalid_clusters = invalid_clusters-set([last_cluster_i])
            
        valid_clusters = list(set(range(0, self.max_topics))-invalid_clusters)
        return valid_clusters
    
    def get_wi_segment(self, wi, u_clusters):
        '''
        Returns the u_cluster for the wi word
        :param wi: index of the word. This index is relative to full document collection
        '''
        for u_cluster in u_clusters:
            if u_cluster.has_word(wi):
                return u_cluster
        return None #Should never happen...
    
    def get_k_segment(self, doc_i, k, u_clusters):
        '''
        Returns the set of word from documents doc_i
        that are in the segment (u_cluster) with topic k
        '''
        target_u_cluster = None
        for u_cluster in u_clusters:
            if u_cluster.k == k:
                target_u_cluster = u_cluster
                break
        if target_u_cluster is None:
            return []
        u_begin, u_end = target_u_cluster.get_segment(doc_i)
        if u_begin is None:
            return []
        words = []
        for u in range(u_begin, u_end+1):
            words += self.data.d_u_wi_indexes[doc_i][u]
        return words
    
    def get_k_cluster(self, k, u_clusters):
        for u_cluster in u_clusters:
            if u_cluster.k == k:
                return u_cluster
        return None
    
    def get_next_cluster(self, k, doc_i, u_clusters):
        cluster_order = self.get_cluster_order(doc_i, u_clusters)
        for i, k2 in enumerate(cluster_order):
            if k2 == k:
                next_cluster_k = cluster_order[i+1]
                break
        next_u_cluster = self.get_k_cluster(next_cluster_k, u_clusters)
        return next_u_cluster
    
    def get_u_segment(self, doc_i, u, u_clusters):
        for u_cluster in u_clusters:
            if u_cluster.has_doc(doc_i):
                u_begin, u_end = u_cluster.get_segment(doc_i)
                if u >= u_begin and u <= u_end:
                    return u_cluster.k, u_begin, u_end
                
    def get_doc_i_clusters(self, doc_i, u_clusters):
        k_list = []
        for u_cluster in u_clusters:
            if u_cluster.has_doc(doc_i):
                k_list.append(u_cluster.k)
        return k_list     
    
    def assign_target_k(self, u_begin, u_end, doc_i, k_target, possible_clusters, u_clusters):
        u_k_target_cluster = self.get_k_cluster(k_target, u_clusters)
        if u_k_target_cluster is not None:
            u_k_target_cluster.add_sents(u_begin, u_end, doc_i)
            if k_target not in possible_clusters:
                u_begin_k_target, u_end_k_target = u_k_target_cluster.get_segment(doc_i)
                for k in range(self.max_topics):
                    if k == k_target:
                        continue
                    
                    u_k_cluster = self.get_k_cluster(k, u_clusters)
                    if u_k_cluster is None:
                        continue
                    
                    if u_k_cluster.has_doc(doc_i):
                        u_begin_di, u_end_di = u_k_cluster.get_segment(doc_i)
                        if u_begin_di > u_begin_k_target:
                            doc_i_word_counts = np.sum(self.data.doc_word_counts(doc_i)[u_begin_di:u_end_di+1], axis=0)
                            u_k_cluster.remove_doc(doc_i, doc_i_word_counts)
                            if len(u_k_cluster.get_docs()) == 0:
                                u_clusters.remove(u_k_cluster)
                            u_k_target_cluster.add_sents(u_begin_di, u_end_di, doc_i)
        else:
            u_k_cluster = SentenceCluster(u_begin, u_end, [doc_i], k_target)
            u_clusters.append(u_k_cluster)
        return u_clusters
    
    def is_cached_seg(self, seg_ll, cached_segs):
        is_cached = False
        for cached_seg_ll, cached_u_clusters in cached_segs:
            if cached_seg_ll == seg_ll:
                is_cached = True
                break
            if cached_seg_ll < seg_ll:
                is_cached = False
                break
        return is_cached
    
    def segment_log_prior(self, seg_size):
        seg_prob = norm(self.seg_dur, self.std).pdf(seg_size)
        return np.log(seg_prob)
        
    def segmentation_log_prior(self, u_clusters):
        log_prior = 0.0
        for doc_i in range(self.data.n_docs):
            doc_i_segmentation = self.get_segmentation(doc_i, u_clusters)
            if len(doc_i_segmentation) == 0:
                continue
            doc_i_segmentation[-1] = 1
            seg_size = 0
            for u in doc_i_segmentation:
                if u == 1:
                    log_prior += self.segment_log_prior(seg_size)
                    seg_size = 0
                else:
                    seg_size += 1
        return log_prior
    
    def segment_ll(self, word_counts):
        '''
        Returns the likelihood if we considering all sentences (word_counts)
        as a single language model.
        :param seg_word_counts: vector with the size equal to the length of
        the vocabulary and values with the corresponding word counts.
        '''
        f1 = gammaln(word_counts+self.beta).sum()
        f2 = gammaln((word_counts+self.beta).sum())
        seg_ll = self.seg_ll_C+f1-f2
        return seg_ll
    
    def segmentation_ll(self, u_clusters):
        '''
        Returns the log likelihood of the segmentation of all documents.
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        segmentation_ll = 0.0
        for u_cluster in u_clusters:
            word_counts = u_cluster.get_word_counts()
            segmentation_ll += self.segment_ll(word_counts)
            
        if self.use_prior:
            segmentation_ll += self.segmentation_log_prior(u_clusters)
                                
        return segmentation_ll
    
    def dp_segmentation_step(self):
        with open(self.log_dir+"dp_tracker_"+self.desc+".txt", "a+") as f:
            f.write("DP tracking:\n")
            for u_end in range(self.data.max_doc_len):
                f.write("Tracking line %d\n"%(u_end))
                if u_end == 14:
                    a = 0
                best_seg_ll = -np.inf
                best_seg_clusters = None
                best_u_begin = -1
                for u_begin in range(u_end+1):
                    if u_begin == 4:
                        a = 0
                    seg_ll, seg_clusters = self.seg_func(u_begin, u_end)
                    if seg_ll > best_seg_ll:
                        best_seg_ll = seg_ll
                        best_seg_clusters = seg_clusters
                        best_u_begin = u_begin
                    f.write("(%d,%d)\tll: %.3f\n"%(u_begin, u_end, seg_ll))
                    for doc_i in range(self.data.n_docs):
                        f.write(str(self.get_segmentation(doc_i, seg_clusters))+" "
                                +str(self.get_seg_with_topics(doc_i, seg_clusters))+"\n")
                    f.write("\n")
                f.write("============\n")
                self.best_segmentation[u_end] = best_seg_clusters
                #self.print_seg(best_seg_clusters)
            #print("==========================")
        
class Data(object):
    '''
    Wrapper class for MultiDocument object. Represent the full collection of documents.
    In this segmentor implementation it is convenient to have
    individual word counts for each document. 
    '''
    def __init__(self, docs):
        self.doc_synth = docs
        self.docs_rho_gs = [doc.rho for doc in docs.get_single_docs()]
        self.doc_rho_topics = docs.doc_rho_topics
        self.docs_index = docs.docs_index
        self.W = docs.W
        self.W_I_words = docs.W_I_words #Vector the vocabulary indexes of ith word in the full collection
        self.d_u_wi_indexes = docs.d_u_wi_indexes #Contains the a list of word indexes organizes by document and by sentence
        self.n_docs = docs.n_docs
        self.doc_lens = []
        self.docs_word_counts = []
        self.multi_doc_slicer(docs)
        self.max_doc_len = np.max(self.doc_lens)
        self.total_sents = 0
        self.total_words = 0 #Number of words in full document collection
        for doc_i in range(self.n_docs):
            self.total_words += np.sum(self.doc_word_counts(doc_i))
            self.total_sents += self.doc_len(doc_i)
        
    def multi_doc_slicer(self, docs):
        doc_begin = 0
        for doc_end in docs.docs_index:
            doc = copy.deepcopy(docs)
            self.doc_lens.append(doc_end - doc_begin)
            U_W_counts = doc.U_W_counts[doc_begin:doc_end, :]
            self.docs_word_counts.append(U_W_counts)
            doc_begin = doc_end
        
    def doc_len(self, doc_i):
        '''
        Returns the length (number of sentences) of doc_i
        :param doc_i: document index
        '''
        return self.doc_lens[doc_i]
        
    def doc_word_counts(self, doc_i):
        '''
        Returns the word count matrix for doc_i
        :param doc_i: document index
        '''
        return self.docs_word_counts[doc_i]
    
    def get_doc_i(self, u):
        for doc_i, doc_i_index in enumerate(self.docs_index):
            if u < doc_i_index:
                return doc_i
        
    def get_rho_u_clusters(self):
        gs_u_clusters = []
        doc_i = 0
        i = 0
        doc_u = 0
        for u in range(len(self.doc_synth.rho)):
            k = self.doc_synth.doc_topic_seq[doc_i][i]
            u_k_cluster = None
            for u_cluster in gs_u_clusters:
                if u_cluster.k == k:
                    u_k_cluster = u_cluster
                    break
            if u_k_cluster is None:
                u_k_cluster = SentenceCluster(doc_u, doc_u, [doc_i], k)
                gs_u_clusters.append(u_k_cluster)
            else:
                u_k_cluster.add_sents(doc_u, doc_u, doc_i)
                
            if self.doc_synth.rho[u] == 1:
                i += 1
                
            if u+1 in self.docs_index:
                i = 0
                doc_i += 1
                doc_u = 0
            else:
                doc_u += 1
        return gs_u_clusters
    
class SentenceCluster(object):
    '''
    Class to keep track of a set of sentences (possibly from different documents)
    that belong to the same segment.
    '''
    def __init__(self, u_begin, u_end, docs, k):
        self.k = k
        self.doc_segs_dict = {}
        self.word_counts = np.zeros(GL_DATA.W)
        
        for doc_i in docs:
            doc_i_len = GL_DATA.doc_len(doc_i)
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            
            if u_end > doc_i_len-1:
                u_end_true = doc_i_len-1
            else:
                u_end_true = u_end
            self.doc_segs_dict[doc_i] = [u_begin, u_end_true]
            self.word_counts += np.sum(GL_DATA.doc_word_counts(doc_i)[u_begin:u_end_true+1], axis=0)
        
        self.wi_list = []
        for doc_i in docs:
            doc_i_len = GL_DATA.doc_len(doc_i)
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            if u_end > doc_i_len-1:
                true_u_end = doc_i_len-1
            else:
                true_u_end = u_end
            for u in range(u_begin, true_u_end+1):
                d_u_words = GL_DATA.d_u_wi_indexes[doc_i][u]
                self.wi_list += d_u_words
                    
    def set_k(self, k):
        self.k = k
        
    def has_word(self, wi):
        return wi in self.wi_list
        
    def has_doc(self, doc_i):
        return doc_i in self.doc_segs_dict.keys()
    
    def add_sents(self, u_begin, u_end, doc_i):
        global GL_DATA
        doc_i_len = GL_DATA.doc_len(doc_i)
        #Accounting for documents with different lengths
        if u_begin > doc_i_len-1:
            return
        if u_end > doc_i_len-1:
            u_end = doc_i_len-1
            
        if self.has_doc(doc_i):
            current_u_begin, current_u_end = self.get_segment(doc_i)
            if u_begin < current_u_begin:
                current_u_begin = u_begin
            if u_end > current_u_end:
                current_u_end = u_end
            self.doc_segs_dict[doc_i] = [current_u_begin, current_u_end]
        else:
            self.doc_segs_dict[doc_i] = [u_begin, u_end]
            
        seg = list(range(u_begin, u_end+1))
        self.word_counts += np.sum(GL_DATA.doc_word_counts(doc_i)[u_begin:u_end+1], axis=0)
        
        for u in seg:
            d_u_words = GL_DATA.d_u_wi_indexes[doc_i][u]
            self.wi_list += d_u_words
            
    def remove_doc(self, doc_i, doc_i_word_counts):
        self.word_counts -= doc_i_word_counts
        for doc_w_i in self.get_doc_words(doc_i):
            self.wi_list.remove(doc_w_i)
            
        self.doc_segs_dict.pop(doc_i)
        
    def remove_seg(self, doc_i, u_begin, u):
        current_seg = self.doc_segs_dict[doc_i]
        if current_seg[0] == u_begin and current_seg[1] == u:
            self.doc_segs_dict.pop(doc_i)
        else:
            if current_seg[0] >= u_begin:
                self.doc_segs_dict[doc_i] = [u+1, current_seg[1]]
            else:
                self.doc_segs_dict[doc_i] = [current_seg[0], u_begin-1]
            
        self.word_counts -= np.sum(GL_DATA.doc_word_counts(doc_i)[u_begin:u+1], axis=0)
        seg = list(range(u_begin, u+1))
        for u in seg:
            d_u_words = GL_DATA.d_u_wi_indexes[doc_i][u]
            for doc_w_i in d_u_words:
                self.wi_list.remove(doc_w_i)
            
    def get_docs(self):
        return self.doc_segs_dict.keys()
        
    def get_word_counts(self):
        return self.word_counts
        
    def get_words(self):
        return self.wi_list
    
    def get_doc_words(self, doc_i):
        wi_list = []
        u_begin, u_end = self.get_segment(doc_i)
        for u in range(u_begin, u_end+1):
            wi_list += GL_DATA.d_u_wi_indexes[doc_i][u]
        return wi_list
    
    def get_words_minus_doc(self, doc_i):
        wi_list = []
        for doc_j in self.doc_segs_dict:
            if doc_j != doc_i:
                u_begin, u_end = self.get_segment(doc_j)
                for u in range(u_begin, u_end+1):
                    wi_list += GL_DATA.d_u_wi_indexes[doc_j][u]
        return wi_list
    
    def get_segment(self, doc_i):
        '''
        Returns the first and last sentences (u_begin, u_end) of the doc_i
        document in this u_cluster 
        :param doc_i: index of the document
        '''
        seg_bound = self.doc_segs_dict[doc_i]
        u_begin = seg_bound[0]
        u_end = seg_bound[1]
        return u_begin, u_end