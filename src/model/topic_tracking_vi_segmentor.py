'''
Created on Nov 1, 2017

@author: pjdrm
'''
import numpy as np
import copy
from tqdm import trange
from dataset.synthetic_doc_cvb import CVBSynDoc, CVBSynDoc2, CVBSynDoc3
from scipy.special import gammaln
import eval.eval_tools as eval_tools
from itertools import chain, combinations
import time
import toyplot
import toyplot.pdf

SEG_FAST = "fast"
SEG_ALL_COMBS = "all_combs"

class TopicTrackingVIModel(object):

    def __init__(self, beta, data, seg_type=None):
        self.beta = beta
        self.C_beta = np.sum(self.beta)
        self.W = data.W
        self.data = data
        self.best_segmentation = [[] for i in range(self.data.max_doc_len)]
        self.seg_ll_C = gammaln(self.beta.sum())-gammaln(self.beta).sum()
        #List of matrices (one for each topic). Lines are words in the document collection and columns the vocabulary indexes.
        #The entries contains the value of the corresponding variational parameter.
        self.qz = self.init_variational_params(self.data.total_words, self.data.W, self.data.max_doc_len, self.data.W_I_words) 
        
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
    
    def print_seg(self, u_clusters):
        print("==========================")
        for doc_i in range(self.data.n_docs):
            seg = self.get_segmentation(doc_i, u_clusters)
            print("Doc %d: %s" % (doc_i, str(seg)))
            
    def get_segmentation(self, doc_i, u_clusters):
        '''
        Returns the final segmentation for a document.
        This is done by backtracking the best segmentations
        in a bottom-up fashion.
        :param doc_i: document index
        '''
        hyp_seg = []
        for u_cluster in u_clusters:
            found_doc = False
            for u, doc_j in zip(u_cluster.u_list, u_cluster.doc_list):
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
            all_segs += self.get_segmentation(doc_i, self.best_segmentation[-1])
        return all_segs
            
    def get_last_cluster(self, doc_i, u_clusters):
        '''
        Returns the last cluster index where doc_i is present
        :param doc_i: document index
        :param u_clusters: list of sentence cluster corresponding to a segmentation
        '''
        found_doc = False
        for cluster_i, u_cluster in enumerate(u_clusters):
            if u_cluster.has_doc(doc_i):
                if not found_doc:
                    found_doc = True
            elif found_doc:
                return cluster_i-1
        return len(u_clusters)-1 #Case where the last cluster was the last one in the list
    
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
        for u_cluster in u_clusters:
            if u_cluster.k == k:
                target_u_cluster = u_cluster
                break
        u_begin, u_end = target_u_cluster.get_segment(doc_i)
        words = []
        for u in range(u_begin, u_end+1):
            words += self.data.d_u_wi_indexes[doc_i][u]
        return words
    
    def qz_words_minus_wi(self, doc_i, wi, k, u_clusters):
        '''
        Returns the set of words that influence the qz update
        of wi.
        '''
        wi_segment = self.get_wi_segment(wi, u_clusters)
        words = wi_segment.get_words()-wi
        if k == wi_segment.k:
            return words
        else:
            k_diff = k-wi_segment.k
            if k_diff < 0:
                #Case we need to merge with the segments in front
                sign = 1
            else:
                #Case we need to merge with the segments behind
                sign = -1
            for k_btw in range(np.abs(k_diff)):
                words += self.get_k_segment(doc_i, k+k_btw*sign)
            return words
            
    def var_update_k_val(self, doc_i, wi, k, u_clusters):
        '''
        Return the value of the numerator for the variational update expression.
        :param doc_i: document index
        :param wi: word index (relative to the full collection of documents)
        :param k: topic/language model/segment index
        :param u_clusters: list of sentence clusters representing a segmentation of all documents
        '''
        words_update = self.qz_words_minus_wi(doc_i, wi, k, u_clusters)
        E_counts_f2 = self.qz[words_update, k]
        Var_counts_f2 = np.sum(E_counts_f2*(1.0-E_counts_f2))
        C_beta_E_counts_f2_sum = np.sum(self.C_beta+E_counts_f2)
        E_q_f2 = np.log(C_beta_E_counts_f2_sum)-(Var_counts_f2/(2.0*(C_beta_E_counts_f2_sum)**2))
        
        
        word_mask = (self.data.W_I_words[words_update]==self.data.W_I_words[wi]).astype(np.int)
        E_counts_f1 = E_counts_f2*word_mask
        Var_counts_f1 = np.sum(E_counts_f1*(1.0-E_counts_f1))
        C_beta_E_counts_f1_sum = np.sum(self.C_beta+E_counts_f1)
        E_q_f1 = np.log(C_beta_E_counts_f1_sum)-(Var_counts_f1/(2.0*(C_beta_E_counts_f1_sum)**2))
        num = np.exp(E_q_f1-E_q_f2)
        
        return num
        
    def var_param_update(self, doc_i, wi, k, u_clusters):
        '''
        Updates the variational parameter of word wi for topic k.
        :param doc_i: document index
        :param wi: word index (relative to the full collection of documents)
        :param k: topic/language model/segment index
        :param u_clusters: list of sentence clusters representing a segmentation of all documents
        '''
        num_k = self.var_update_k_val(doc_i, wi, k, u_clusters)
        denom = num_k
        for k_denom in range(self.data.max_doc_len): #TODO: seems very inefficient, would like to do it in a single matrix operation
            if k_denom == k:
                continue
            denom += self.var_update_k_val(doc_i, wi, k_denom, u_clusters)
        denom = np.exp(denom)
        return num_k/denom
    
    def variational_step(self, u_clusters):
        '''
        Update the variational parameters for all words and all topics.
        :param u_clusters: list of sentence clusters representing the current segmentation state 
        '''
        for k in range(self.data.max_doc_len):
            for doc_i in range(self.data.n_docs):
                for u in self.data.d_u_wi_indexes[doc_i]:
                    for wi in self.data.d_u_wi_indexes[doc_i][u]:
                        self.var_param_update(doc_i, wi, k, u_clusters)
    
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
            qz_counts = np.sum(self.qz[u_cluster.k][u_cluster.wi_list], axis=0)
            segmentation_ll += self.segment_ll(qz_counts)
        return segmentation_ll
    
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
                if u_cluster.k == cluster_i+n_skips+1:
                    target_cluster = u_cluster
                    break
            if target_cluster is not None: #The language model corresponding to this cluster might already exists due to other documents having different segmentation at this stage
                target_cluster.add_sents(u_begin, u_end, doc_i)
            else:
                new_cluster = SentenceCluster(u_begin, u_end, [doc_i], self.data, cluster_i+n_skips+1)
                u_clusters.append(new_cluster)
                
    def segment_u(self, u_begin, u_end):
        '''
        Estimates, for all documents, the best segmentation
        from u_end to u_begin (column index in the DP matrix).
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        if u_begin == 0:#The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            u_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data)
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
        if u_begin == 0: #The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            u_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data, 0)
            segmentation_ll = self.segmentation_ll([u_cluster])
            return segmentation_ll, [u_cluster]
        
        best_seg = self.best_segmentation[u_begin-1]
        final_seg_ll = None
        for doc_i in range(self.data.n_docs):
            seg_fit_prev = copy.deepcopy(best_seg)
            self.fit_sentences(u_begin, u_end, [doc_i], seg_fit_prev) #Note that this changes seg_fit_prev
            seg_fit_prev_ll = self.segmentation_ll(seg_fit_prev) #Case where we did not open a "new" segment
            
            seg_new_seg = copy.deepcopy(best_seg)
            self.new_seg_point(u_begin, u_end, [doc_i], seg_new_seg, n_skips=0) #Note that this changes seg_new_seg
            seg_new_ll = self.segmentation_ll(seg_new_seg) #Case where we opened a "new" segment
            
            seg_new_seg_skip = copy.deepcopy(best_seg)
            self.new_seg_point(u_begin, u_end, [doc_i], seg_new_seg_skip, n_skips=1) #Note that this changes seg_new_seg_skip
            seg_new_skip_ll = self.segmentation_ll(seg_new_seg_skip) #Case where we opened a "new" segment and skip a topic
            
            seg_ll_list = [seg_fit_prev_ll, seg_new_ll, seg_new_skip_ll]
            max_ll = max(seg_ll_list)
            max_index = seg_ll_list.index(max_ll)
            
            if max_index == 0:
                best_seg = seg_fit_prev
                final_seg_ll = seg_fit_prev_ll
            elif max_index == 1:
                best_seg = seg_new_seg
                final_seg_ll = seg_new_ll
            else:
                best_seg = seg_new_seg_skip
                final_seg_ll = seg_new_skip_ll
                
        return final_seg_ll, best_seg
            
    def dp_segmentation_step(self):
        t = trange(self.data.max_doc_len, desc='', leave=True)
        for u_end in t:
            best_seg_ll = -np.inf
            best_seg_clusters = None
            for u_begin in range(u_end+1):
                seg_ll, seg_clusters = self.seg_func(u_begin, u_end)
                if seg_ll > best_seg_ll:
                    best_seg_ll = seg_ll
                    best_seg_clusters = seg_clusters
                t.set_description("Matrix: (%d,%d)" % (u_end, u_begin))
            self.best_segmentation[u_end] = best_seg_clusters
            #self.print_seg(best_seg_clusters)
        #print("==========================")
        
    def segment_docs(self, n_iters=3): #TODO: check if the segmentation changes and use that as criteria for stopping
        '''
        Segments the full collection of documents. Alternates between using a Dynamic Programming
        procedure for segmentation and performing variational inference to update the
        certainty about words belonging to a topic (segments/language model).
        :param n_iters: number of iterations to perform
        '''
        for i in range(n_iters):
            self.dp_segmentation_step()
            best_segmetnation = self.best_segmentation[-1]
            self.variational_step(best_segmetnation)
    
class Data(object):
    '''
    Wrapper class for MultiDocument object. Represent the full collection of documents.
    In this segmentor implementation it is convenient to have
    individual word counts for each document. 
    '''
    def __init__(self, docs):
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
        
        self.wi_list = []
        for doc_i, u in zip(self.doc_list, self.u_list):
            d_u_words = self.data.d_u_wi_indexes[doc_i][u]
            self.wi_list += d_u_words
    
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
            
        seg = list(range(u_begin, u_end+1))
        seg_len =  u_end-u_begin+1
        self.u_list += seg
        self.doc_list += [doc_i]*seg_len
        
        for u in seg:
            d_u_words = self.data.d_u_wi_indexes[doc_i][u]
            self.wi_list += d_u_words
        
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

def print_segmentation(seg_desc, seg_results):
    for seg in seg_results:
        print("%s: %s" % (seg_desc, str(seg)))
        
def single_vs_md_eval(doc_synth, beta, md_all_combs=True, md_fast=True, print_flag=False):
    '''
    Print the WD results when segmenting single documents
    and all of them simultaneously (multi-doc model)
    :param doc_synth: collection of synthetic documents
    :param beta: beta prior vector
    :param print_flag: boolean to print or not the segmentation results
    '''
    single_docs = doc_synth.get_single_docs()
    single_doc_wd = []
    time_wd_results = []
    start = time.time()
    sd_segs = []
    for doc in single_docs:
        data = Data(doc)
        vi_tt_model = TopicTrackingVIModel(beta, data)
        vi_tt_model.dp_segmentation_step()
        sd_segs.append(vi_tt_model.get_segmentation(0, vi_tt_model.best_segmentation[-1]))
        single_doc_wd += eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc)
    end = time.time()
    sd_time = (end - start)
    #single_doc_wd = ['%.3f' % wd for wd in single_doc_wd]
    time_wd_results.append(("SD", sd_time, ['%.3f' % wd for wd in single_doc_wd]))
    
    data = Data(doc_synth)
    if md_all_combs:
        vi_tt_model = TopicTrackingVIModel(beta, data, seg_type=SEG_ALL_COMBS)
        start = time.time()
        vi_tt_model.dp_segmentation_step()
        end = time.time()
        md_time = (end - start)
        multi_doc_wd = eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth)
        #multi_doc_wd = ['%.3f' % wd for wd in multi_doc_wd]
        time_wd_results.append(("MD", md_time, ['%.3f' % wd for wd in multi_doc_wd]))
        
        md_segs = []
        for doc_i in range(vi_tt_model.data.n_docs):
            md_segs.append(vi_tt_model.get_segmentation(doc_i, vi_tt_model.best_segmentation[-1]))
    
    if md_fast: 
        md_fast_segs = []
        vi_tt_model = TopicTrackingVIModel(beta, data, seg_type=SEG_FAST)
        start = time.time()
        vi_tt_model.dp_segmentation_step()
        end = time.time()
        md_fast_time = (end - start)
        multi_fast_doc_wd = eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth)
        time_wd_results.append(("MF", md_fast_time, ['%.3f' % wd for wd in multi_fast_doc_wd]))
        for doc_i in range(vi_tt_model.data.n_docs):
            md_fast_segs.append(vi_tt_model.get_segmentation(doc_i, vi_tt_model.best_segmentation[-1]))
            
    if print_flag:
        gs_segs = []
        for gs_doc in doc_synth.get_single_docs():
            gs_segs.append(gs_doc.rho.tolist())
            
        if md_all_combs and md_fast:
            for doc_i in range(doc_synth.n_docs):
                print("%s: %s" % ("GS", str(gs_segs[doc_i])))
                print("%s: %s" % ("SD", str(sd_segs[doc_i])))
                print("%s: %s" % ("MD", str(md_segs[doc_i])))
                print("%s: %s\n" % ("MF", str(md_fast_segs[doc_i])))
        elif md_all_combs:
            for doc_i in range(doc_synth.n_docs):
                print("%s: %s" % ("GS", str(gs_segs[doc_i])))
                print("%s: %s" % ("SD", str(sd_segs[doc_i])))
                print("%s: %s\n" % ("MD", str(md_segs[doc_i])))
        else:
            for doc_i in range(doc_synth.n_docs):
                print("%s: %s" % ("GS", str(gs_segs[doc_i])))
                print("%s: %s" % ("SD", str(sd_segs[doc_i])))
                print("%s: %s\n" % ("MF", str(md_fast_segs[doc_i])))
    for time_res in time_wd_results:  
        print("%s: %s time: %f" % (time_res[0], time_res[2], time_res[1]))
    
    return single_doc_wd, multi_fast_doc_wd
    
def md_eval(doc_synth, beta):
    vi_tt_model = TopicTrackingVIModel(beta, data, seg_type=SEG_FAST)
    start = time.time()
    vi_tt_model.segment_docs()
    end = time.time()
    seg_time = (end - start)
    md_segs = []
    for doc_i in range(vi_tt_model.data.n_docs):
        md_segs.append(vi_tt_model.get_segmentation(doc_i, vi_tt_model.best_segmentation[-1]))
            
    gs_segs = []
    for gs_doc in doc_synth.get_single_docs():
        gs_segs.append(gs_doc.rho)
        
    for md_seg, gs_seg in zip(md_segs, gs_segs):
        print("GS: " + str(gs_seg.tolist()))
        print("MD: " + str(md_seg)+"\n")
    print("Time: %f" % seg_time)
        
    print(eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth))
    
def merge_docs(target_docs):
    target_docs_copy = copy.deepcopy(target_docs)
    merged_doc = target_docs_copy[0]
    all_rho = []
    all_docs_index = []
    all_U_W_counts = None
    carry_index = 0
    for doc_synth in target_docs:
        new_index = (np.array(doc_synth.docs_index)+carry_index).tolist()
        carry_index = new_index[-1]
        all_docs_index += new_index
        doc_synth.rho[-1] = 1
        all_rho += doc_synth.rho.tolist()
        if all_U_W_counts is None:
            all_U_W_counts = doc_synth.U_W_counts
        else:
            all_U_W_counts = np.vstack((all_U_W_counts, doc_synth.U_W_counts))
            
    all_rho[-1] = 0
    
    merged_doc.n_docs = len(target_docs)
    merged_doc.rho = np.array(all_rho)
    merged_doc.docs_index = all_docs_index
    merged_doc.U_W_counts = all_U_W_counts
    merged_doc.isMD = True
    
    return merged_doc
    
def incremental_eval(doc_synth, beta):
    def grouped_bars(axes, data, group_names, group_width=None):
        if group_width is None:
            group_width=1 - 1.0 / (data.shape[1] + 1)
            
        group_left_edges = np.arange(data.shape[0], dtype="float") - (group_width / 2.0)
        bar_width = group_width / data.shape[1]
        
        marks = []
        axes.x.ticks.locator = toyplot.locator.Explicit(labels=group_names)
        for index, series in enumerate(data.T):
            left_edges = group_left_edges + (index * bar_width)
            right_edges = group_left_edges + ((index + 1) * bar_width)
            marks.append(axes.bars(left_edges, right_edges, series, opacity=0.5))
            
        return marks

    single_docs = doc_synth.get_single_docs()
    all_sd_results = []
    all_md_results = []
    for i in range(1, doc_synth.n_docs+1):
        target_docs = single_docs[:i]
        merged_doc_synth = merge_docs(target_docs)
        sd_results, md_results = single_vs_md_eval(merged_doc_synth, beta, md_all_combs=False, print_flag=True)
        all_sd_results.append(sd_results)
        all_md_results.append(md_results)
        
    final_results = []
    for sd_wds, mf_wds in zip(all_sd_results, all_md_results):
        n_ties = 0.0
        n_mf_win = 0.0
        n_mf_lost = 0.0
        for sd_wd, mf_wd in zip(sd_wds, mf_wds):
            if sd_wd == mf_wd:
                n_ties += 1.0
            elif mf_wd > sd_wd:
                n_mf_win += 1.0
            else:
                n_mf_lost += 1.0
        n_total = n_ties+n_mf_win+n_mf_lost
        n_ties_percent = n_ties/n_total
        n_mf_win_percent = n_mf_win/n_total
        n_mf_lost_percent = n_mf_lost/n_total
        final_results.append(np.array([n_ties_percent, n_mf_win_percent, n_mf_lost_percent])*100.0)
        
    group_names = list(range(1, doc_synth.n_docs+1))
    canvas = toyplot.Canvas(width=600, height=300)
    axes = canvas.cartesian()
    axes.x.label.text = "#Docs"
    axes.y.label.text = "Percentage"
    marks = grouped_bars(axes, np.array(final_results), group_names)
    canvas.legend([
    ("Tie", marks[0]),
    ("Lose", marks[1]),
    ("Win", marks[2])
    ],
    corner=("top-right", 0, 100, 50),
    );
    toyplot.pdf.render(canvas, "incremental_eval_results.pdf")
    
    
    
W = 10
beta = np.array([0.1]*W)
n_docs = 2
doc_len = 20
pi = 0.25
sent_len = 8
#doc_synth = CVBSynDoc(beta, pi, sent_len, doc_len, n_docs)
n_seg = 2
doc_synth = CVBSynDoc2(beta, pi, sent_len, n_seg, n_docs)
#doc_synth = CVBSynDoc3(beta)
data = Data(doc_synth)

#incremental_eval(doc_synth, beta)
#single_vs_md_eval(doc_synth, beta, md_all_combs=False , md_fast=True, print_flag=True)
#single_vs_md_eval(doc_synth, beta, md_all_combs=False , md_fast=True, print_flag=True)
md_eval(doc_synth, beta)
