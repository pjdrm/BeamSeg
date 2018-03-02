'''
Created on Feb 22, 2018

@author: pjdrm
'''
from model.dp.segmentor import AbstractSegmentor, SentenceCluster
from itertools import chain, combinations
import copy
import numpy as np

SEG_FAST = "fast"
SEG_ALL_COMBS = "all_combs"

class MultiDocDPSeg(AbstractSegmentor):
    
    def __init__(self, beta, data, seg_type=None):
        super(MultiDocDPSeg, self).__init__(beta, data)
        
        if seg_type is None or seg_type == SEG_ALL_COMBS:
            self.seg_func = self.segment_u
            self.doc_combs_list = self.init_doc_combs()#All n possible combinations (up to the number of documents). Its a list of pairs where the first element is the combination and second the remaining docs
        elif seg_type == SEG_FAST:
            self.seg_func = self.segment_u_fast
        else:
            raise Exception("ERROR: unknown seg_type")
        
    def init_doc_combs(self):
        all_docs = set(range(self.data.n_docs))
        doc_combs = chain.from_iterable(combinations(all_docs, r) for r in range(1,len(all_docs)+1))
        doc_combs_list = []
        for doc_comb in doc_combs:
            other_docs = all_docs - set(doc_comb)
            doc_combs_list.append([doc_comb, other_docs])
        return doc_combs_list
    
    def fit_sentences(self, u_begin, u_end, docs, u_clusters):
        '''
        Adds the u sentences (a segment) of the docs to the best u_cluster. That is,
        the cluster where u-1 (of the corresponding doc) is located. Note
        that the cluster is for the best likelihood, thus, we need to add
        u to the same cluster as u-1.
        :param seg_u_list: list of u indexes corresponding to a segment
        :param docs: list of documents
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        for doc in docs:
            for u_cluster in reversed(u_clusters):#We reverse because we are only checking the document list of the cluster
                if u_cluster.has_doc(doc):
                    u_cluster.add_sents(u_begin, u_end, doc)
                    break
    
    def new_seg_point(self, u_begin, u_end, doc_comb, u_clusters, n_skips=0):
        '''
        Considers the segment u_begin to u_end as a new segmentation points for all
        documents in doc_comb. The sentences are added to the corresponding cluster
        (a new cluster is generated if necessary).
        :param u_begin: beginning sentence index
        :param u_end: end sentence index
        :param doc_comb: list of document indexes
        :param u_clusters: list of sentence cluster corresponding to a segmentation
        :param n_skips: number of topics to skip when adding the new segmentation point
        '''
        for doc_i in doc_comb:
            cluster_i = self.get_last_cluster(doc_i, u_clusters)
            target_cluster = None #For new segmentation points the target cluster is the last cluster (topic) where doc_i appear +1
            for u_cluster in u_clusters: #Checking if the target cluster already exists
                if u_cluster.k == u_clusters[cluster_i].k+n_skips+1:
                    target_cluster = u_cluster
                    break
            if target_cluster is not None: #The language model corresponding to this cluster might already exists due to other documents having different segmentation at this stage
                target_cluster.add_sents(u_begin, u_end, doc_i)
            else:
                new_cluster = SentenceCluster(u_begin, u_end, [doc_i], self.data, u_clusters[cluster_i].k+n_skips+1)
                u_clusters.append(new_cluster)
    
    def segmentation_ll(self, u_clusters):
        '''
        Returns the log likelihood of the segmentation of all documents.
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        segmentation_ll = 0.0
        for u_cluster in u_clusters:
            qz_counts = u_cluster.get_word_counts() #Version that does not use qz weight at all
            segmentation_ll += self.segment_ll(qz_counts)
        return segmentation_ll
    
    def segment_u(self, u_begin, u_end):
        '''
        Estimates, for all documents, the best segmentation
        from u_end to u_begin (column index in the DP matrix).
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        if u_begin == 0:#The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            u_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data, 0)
            segmentation_ll = self.segmentation_ll([u_cluster])
            return segmentation_ll, [u_cluster]
           
        best_seg_ll = -np.inf
        best_seg_clusters = None
        for doc_comb, other_docs in self.doc_combs_list:
            best_seg = copy.deepcopy(self.best_segmentation[u_begin-1])
            self.fit_sentences(u_begin, u_end, other_docs, best_seg) #Note that this changes best_seg
            self.new_seg_point(u_begin, u_end, doc_comb, best_seg) #Note that this changes best_seg
            #if self.valid_segmentation(best_seg):
            segmentation_ll = self.segmentation_ll(best_seg)
            if segmentation_ll >= best_seg_ll:
                best_seg_ll = segmentation_ll
                best_seg_clusters = best_seg
        return best_seg_ll, best_seg_clusters
    
    def segment_u_fast(self, u_begin, u_end):
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
        if u_begin == 0:#The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            u_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data, 0)
            segmentation_ll = self.segment_ll(u_cluster.get_word_counts())
            return segmentation_ll, [u_cluster]
        
        best_seg = self.best_segmentation[u_begin-1]
        final_seg_ll = None
        for doc_i in range(self.data.n_docs):
            seg_fit_prev = copy.deepcopy(best_seg)
            self.fit_sentences(u_begin, u_end, [doc_i], seg_fit_prev) #Note that this changes seg_fit_prev
            seg_fit_prev_ll = self.segmentation_ll(seg_fit_prev) #Case where we did not open a "new" segment
            
            seg_new_seg = copy.deepcopy(best_seg)
            self.new_seg_point(u_begin, u_end, [doc_i], seg_new_seg) #Note that this changes seg_new_seg
            seg_new_ll = self.segmentation_ll(seg_new_seg) #Case where we opened a "new" segment
            
            if seg_fit_prev_ll > seg_new_ll:
                best_seg = seg_fit_prev
                final_seg_ll = seg_fit_prev_ll
            else:
                best_seg = seg_new_seg
                final_seg_ll = seg_new_ll
                
        return final_seg_ll, best_seg
    
    def segment_docs(self):
        self.dp_segmentation_step()