'''
Created on Feb 22, 2018

@author: pjdrm
'''
import numpy as np
import copy
from scipy.special import gammaln
import os
import collections

class AbstractSegmentor(object):

    def __init__(self, beta, data, max_topics=None, log_dir="../logs/", desc="Abs_seg"):
        self.data = data
        self.max_topics = self.data.max_doc_len if max_topics is None else max_topics
        self.desc = desc
        self.log_dir = log_dir
        self.beta = beta
        self.C_beta = np.sum(self.beta)
        self.W = data.W
        self.best_segmentation = [[] for i in range(self.data.max_doc_len)]
        self.seg_ll_C = gammaln(self.beta.sum())-gammaln(self.beta).sum()
        
        os.remove(self.log_dir+"dp_tracker_"+self.desc+".txt") if os.path.exists(self.log_dir+"dp_tracker_"+self.desc+".txt") else None

    def print_seg(self, u_clusters):
        print("==========================")
        for doc_i in range(self.data.n_docs):
            seg = self.get_final_segmentation(doc_i, u_clusters)
            print("Doc %d: %s" % (doc_i, str(seg)))
            
    def print_seg_with_topics(self, doc_i, u_clusters):
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
        return str(rho)
            
    def get_cluster_order(self, doc_i, u_clusters):
        cluster_k_list = []
        for u_cluster in u_clusters:
            for u, doc_j in zip(u_cluster.u_list, u_cluster.doc_list):
                if doc_j != doc_i:
                    continue
                cluster_k_list.append([u_cluster.k, u])
                break
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
        for u_cluster_k in cluster_order:
            for u_cluster in u_clusters:
                if u_cluster.k != u_cluster_k:
                    continue
                found_doc = False
                for doc_j in u_cluster.doc_list:
                    if doc_j == doc_i:
                        hyp_seg.append(0)
                        found_doc = True
                if found_doc:
                    hyp_seg[-1] = 1
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
                last_sent_u_cluster = -1
                for u, doc_j in zip(u_cluster.u_list, u_cluster.doc_list):
                    if doc_i != doc_j:
                        continue
                    last_sent_u_cluster = u
                if last_sent_u_cluster > last_sent:
                    last_cluster = cluster_i
                    last_sent = last_sent_u_cluster
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
                last_sent_u_cluster = -1
                for u, doc_j in zip(u_cluster.u_list, u_cluster.doc_list):
                    if doc_i != doc_j:
                        continue
                    last_sent_u_cluster = u
                if last_sent_u_cluster > last_sent:
                    last_cluster_i = u_cluster.k
                    last_sent = last_sent_u_cluster
                    
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
        for k_next in range(k+1, self.max_topics):
            for u_cluster in u_clusters:
                if u_cluster.k == k_next and u_cluster.has_doc(doc_i):
                    return u_cluster
        return None
    
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
                                +str(self.print_seg_with_topics(doc_i, seg_clusters))+"\n")
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
        self.docs_index = docs.docs_index
        self.W = docs.W
        self.W_I_words = docs.W_I_words #Vector the vocabulary indexes of ith word in the full collection
        self.d_u_wi_indexes = docs.d_u_wi_indexes #Contains the a list of word indexes organizes by document and by sentence
        self.n_docs = docs.n_docs
        self.doc_lens = []
        self.docs_word_counts = []
        self.multi_doc_slicer(docs)
        self.max_doc_len = np.max(self.doc_lens)
        self.total_words = 0 #Number of words in full document collection
        for doc_i in range(self.n_docs):
            self.total_words += np.sum(self.doc_word_counts(doc_i))
        
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
    
class SentenceCluster(object):
    '''
    Class to keep track of a set of sentences (possibly from different documents)
    that belong to the same segment.
    '''
    def __init__(self, u_begin, u_end, docs, data, k):
        self.k = k
        self.data = data
        self.u_list = []
        self.doc_list = []
        self.word_counts = np.zeros(self.data.W)
        
        for doc_i in docs:
            doc_i_len = self.data.doc_len(doc_i)
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            
            if u_end > doc_i_len-1:
                u_end_true = doc_i_len-1
            else:
                u_end_true = u_end
            seg_len = u_end_true-u_begin+1
            self.u_list += list(range(u_begin, u_end_true+1))
            self.doc_list += [doc_i]*seg_len
            self.word_counts += np.sum(self.data.doc_word_counts(doc_i)[u_begin:u_end_true+1], axis=0)
        
        self.wi_list = []
        for doc_i, u in zip(self.doc_list, self.u_list):
            d_u_words = self.data.d_u_wi_indexes[doc_i][u]
            self.wi_list += d_u_words
    
    def set_k(self, k):
        self.k = k
        
    def has_word(self, wi):
        return wi in self.wi_list
        
    def has_doc(self, doc_i):
        return doc_i in self.doc_list
    
    def add_sents(self, u_begin, u_end, doc_i):
        doc_i_len = self.data.doc_len(doc_i)
        #Accounting for documents with different lengths
        if u_begin > doc_i_len-1:
            return
        if u_end > doc_i_len-1:
            u_end = doc_i_len-1
            
        '''
        seg = list(range(u_begin, u_end+1))
        seg_len =  u_end-u_begin+1
        self.u_list += seg
        self.doc_list += [doc_i]*seg_len
        '''
        
        new_u_list = []
        new_doc_list = []
        added_seg = False
        for doc_j, u in zip(self.doc_list, self.u_list):
            new_u_list.append(u)
            new_doc_list.append(doc_j)
            if doc_j == doc_i and u+1 == u_begin:
                added_seg = True
                for new_u in range(u_begin, u_end+1):
                    new_u_list.append(new_u)
                    new_doc_list.append(doc_i)
                    
        seg = list(range(u_begin, u_end+1))
        if not added_seg:
            seg_len =  u_end-u_begin+1
            new_u_list += seg
            new_doc_list += [doc_i]*seg_len
            
        self.u_list = new_u_list
        self.doc_list = new_doc_list
            
        self.word_counts += np.sum(self.data.doc_word_counts(doc_i)[u_begin:u_end+1], axis=0)
        
        for u in seg:
            d_u_words = self.data.d_u_wi_indexes[doc_i][u]
            self.wi_list += d_u_words
            
    def remove_doc(self, doc_i, doc_i_word_counts):
        self.word_counts -= doc_i_word_counts
        new_u_list = []
        new_doc_list = []
        for doc_j, u in zip(self.doc_list, self.u_list):
            if doc_i != doc_j:
                new_u_list.append(u)
                new_doc_list.append(doc_j)
        self.u_list = new_u_list
        self.doc_list = new_doc_list
            
    def get_docs(self):
        return set(self.doc_list)
        
    def get_word_counts(self):
        return self.word_counts
        
    def get_words(self):
        return self.wi_list
    
    def get_doc_words(self, doc_i):
        wi_list = []
        for doc_j, u in zip(self.doc_list, self.u_list):
            if doc_i == doc_j:
                wi_list += self.data.d_u_wi_indexes[doc_i][u]
        return wi_list
    
    def get_words_minus_doc(self, doc_i):
        wi_list = []
        for doc_j, u in zip(self.doc_list, self.u_list):
            if doc_j != doc_i:
                wi_list += self.data.d_u_wi_indexes[doc_j][u]
        return wi_list
    
    def get_segment(self, doc_i):
        '''
        Returns the first and last sentences (u_begin, u_end) of the doc_i
        document in this u_cluster 
        :param doc_i: index of the document
        '''
        u_begin = None
        u_end  = None
        for doc_j, u in zip(self.doc_list, self.u_list): #TODO: make a more efficient version
            if doc_i == doc_j: #TODO: assuming the sentences are always in order, confirm this
                if u_begin is None:
                    u_begin = u
                u_end = u
            
        return u_begin, u_end