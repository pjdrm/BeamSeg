'''
Created on Feb 22, 2018

@author: pjdrm
'''
from model.dp.segmentor import AbstractSegmentor, SentenceCluster
from itertools import chain, combinations
import copy
import numpy as np
import operator
from tqdm import trange

SEG_FAST = "fast"
SEG_ALL_COMBS = "all_combs"
SEG_SKIP_K = "seg_skip_k"

class MultiDocDPSeg(AbstractSegmentor):
    
    def __init__(self, alpha, data, max_topics=None, seg_type=None, desc="MD_DP_seg"):
        super(MultiDocDPSeg, self).__init__(alpha, data, max_topics=max_topics, desc=desc)
        self.max_cache = 5
        self.max_row_cache = 10
        if seg_type is None or seg_type == SEG_ALL_COMBS:
            self.seg_func = self.segment_u
            self.doc_combs_list = self.init_doc_combs()#All n possible combinations (up to the number of documents). Its a list of pairs where the first element is the combination and second the remaining docs
        elif seg_type == SEG_FAST:
            self.seg_func = self.segment_u_fast
        elif seg_type == SEG_SKIP_K:
            self.seg_func = self.segment_u_skip_topics
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
    
    def get_final_segmentation(self, doc_i):
        u_clusters = self.best_segmentation[-1][0][1]
        hyp_seg = self.get_segmentation(doc_i, u_clusters)
        return hyp_seg
    
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
                new_cluster = SentenceCluster(u_begin, u_end, [doc_i], u_clusters[cluster_i].k+n_skips+1)
                u_clusters.append(new_cluster)
    
    def segmentation_ll(self, u_clusters):
        '''
        Returns the log likelihood of the segmentation of all documents.
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        segmentation_ll = 0.0
        for u_cluster in u_clusters:
            word_counts = u_cluster.get_word_counts()
            segmentation_ll += self.segment_ll(word_counts)
        return segmentation_ll
    
    def segment_u(self, u_begin, u_end, prev_u_clusters):
        '''
        Estimates, for all documents, the best segmentation
        from u_end to u_begin (column index in the DP matrix).
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        if u_begin == 0:#The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            u_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), 0)
            segmentation_ll = self.segmentation_ll([u_cluster])
            return [(segmentation_ll, [u_cluster])]
           
        best_seg_ll = -np.inf
        best_seg_clusters = None
        for doc_comb, other_docs in self.doc_combs_list:
            best_seg = copy.deepcopy(prev_u_clusters[1])
            self.fit_sentences(u_begin, u_end, other_docs, best_seg) #Note that this changes u_clusters
            self.new_seg_point(u_begin, u_end, doc_comb, best_seg) #Note that this changes u_clusters
            #if self.valid_segmentation(u_clusters):
            segmentation_ll = self.segmentation_ll(best_seg)
            if segmentation_ll >= best_seg_ll:
                best_seg_ll = segmentation_ll
                best_seg_clusters = best_seg
        return [(best_seg_ll, best_seg_clusters)]
    
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
    
    def segment_u_skip_topics(self, u_begin, u_end, prev_u_clusters):
        '''
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        if u_begin == 0:
            best_seg = []
            for doc_i in range(self.data.n_docs):
                doc_i_len = self.data.doc_len(doc_i)
                #Accounting for documents with different lengths
                if u_begin > doc_i_len-1:
                    continue
                
                best_seg_ll = -np.inf
                best_clusters = None
                for k in range(self.max_topics):
                    u_clusters = copy.deepcopy(best_seg)
                    u_k_cluster = self.get_k_cluster(k, u_clusters)
                    if u_k_cluster is None:
                        u_k_cluster = SentenceCluster(u_begin, u_end, [doc_i], self.data, k)
                        u_clusters.append(u_k_cluster)
                    else:
                        u_k_cluster.add_sents(u_begin, u_end, doc_i)
                    seg_ll = self.segmentation_ll(u_clusters)
                    if seg_ll > best_seg_ll:
                        best_seg_ll = seg_ll
                        best_clusters = u_clusters
                best_seg = best_clusters
            return [(best_seg_ll, best_seg)]
                        
        else:
            final_u_clusters = [prev_u_clusters]
            for doc_i in range(self.data.n_docs):
                doc_i_len = self.data.doc_len(doc_i)
                #Accounting for documents with different lengths
                if u_begin > doc_i_len-1:
                    continue
                
                current_best_u_clusters = []
                '''
                for u_cluster_ll, u_cluster in final_u_clusters:
                    possible_clusters = self.get_valid_insert_clusters(doc_i, u_cluster)
                    seg_results = []
                    for k_target in range(self.max_topics):
                        best_seg = copy.deepcopy(u_cluster)
                        best_seg = self.assign_target_k(u_begin,u_end, doc_i,\
                                                        k_target, possible_clusters, best_seg)
                            
                        seg_ll = self.segmentation_ll(best_seg)
                        seg_results.append((seg_ll, best_seg))
                        
                    seg_results = sorted(seg_results, key=operator.itemgetter(0), reverse=True)
                    best_seg_ll = seg_results[0][0]
                    current_best_u_clusters.append(seg_results[0])
                    for seg_result in seg_results[1:]:
                        if seg_result[0] == best_seg_ll:
                            continue
                        elif abs(seg_result[0]-best_seg_ll) <= 1.5:
                            current_best_u_clusters.append(seg_result)
                        else:
                            break
                final_u_clusters = current_best_u_clusters[:self.max_cache]
                '''
                for u_cluster_ll, u_cluster in final_u_clusters:
                    possible_clusters = self.get_valid_insert_clusters(doc_i, u_cluster)
                    for k_target in range(self.max_topics):
                        best_seg = copy.deepcopy(u_cluster)
                        best_seg = self.assign_target_k(u_begin,u_end, doc_i,\
                                                        k_target, possible_clusters, best_seg)
                            
                        seg_ll = self.segmentation_ll(best_seg)
                        current_best_u_clusters.append((seg_ll, best_seg))
                        
                current_best_u_clusters = sorted(current_best_u_clusters, key=operator.itemgetter(0), reverse=True)
                cached_segs = []
                for seg_result in current_best_u_clusters:
                    seg_ll = seg_result[0]
                    seg_clusters = seg_result[1]
                    
                    is_cached = self.is_cached_seg(seg_ll, cached_segs)
                    if not is_cached:
                        if len(cached_segs) < self.max_cache:
                            cached_segs.append((seg_ll, seg_clusters))
                            cached_segs = sorted(cached_segs, key=operator.itemgetter(0), reverse=True)
                            
                        elif seg_ll > cached_segs[-1][0]:
                            cached_segs[-1] = (seg_ll, seg_clusters)
                            cached_segs = sorted(cached_segs, key=operator.itemgetter(0), reverse=True)
                final_u_clusters = cached_segs
            
        return final_u_clusters
    
    def dp_segmentation_step_cache(self):
        t = trange(self.data.max_doc_len, desc='', leave=True)
        prev_u_end = -1
        with open(self.log_dir+"dp_tracker_"+self.desc+".txt", "a+") as f:
            f.write("DP tracking:\n")
            for u_end in t:
                f.write("Tracking line %d\n"%(u_end))
                if u_end == 9:
                    a = 0
                best_u_begin = -1
                cached_segs = []
                for u_begin in range(u_end+1):
                    t.set_description("(%d, %d)" % (u_end, u_begin))
                    if u_begin == 4:
                        a = 0
                        
                    if u_begin == 0:
                        best_seg = [(-np.inf, [])]
                    else:
                        best_seg = self.best_segmentation[u_begin-1]
                        
                    for best_seg_i in best_seg:
                        seg_results = self.seg_func(u_begin, u_end, best_seg_i)
                        for seg_result in seg_results:
                            seg_ll = seg_result[0]
                            seg_clusters = seg_result[1]
                            
                            is_cached = self.is_cached_seg(seg_ll, cached_segs)
                            if not is_cached:
                                if len(cached_segs) < self.max_row_cache:
                                    cached_segs.append((seg_ll, seg_clusters))
                                    cached_segs = sorted(cached_segs, key=operator.itemgetter(0), reverse=True)
                                    
                                elif seg_ll > cached_segs[-1][0]:
                                    cached_segs[-1] = (seg_ll, seg_clusters)
                                    cached_segs = sorted(cached_segs, key=operator.itemgetter(0), reverse=True)
                    
                    f.write("(%d,%d)\tll: %.3f\n"%(u_begin, u_end, cached_segs[0][0]))
                    for doc_i in range(self.data.n_docs):
                        f.write(str(self.get_segmentation(doc_i, cached_segs[0][1]))+" "
                                +str(self.get_seg_with_topics(doc_i, cached_segs[0][1]))+"\n")
                    f.write("\n")
                f.write("============\n")
                self.best_segmentation[u_end] = cached_segs
                #self.print_seg(best_seg_clusters)
            #print("==========================")
            
    def segment_docs(self):
        self.set_gl_data(self.data)
        self.dp_segmentation_step_cache()