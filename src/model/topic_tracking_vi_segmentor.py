'''
Created on Nov 1, 2017

@author: pjdrm
'''
import numpy as np
import copy
from tqdm import trange
from dataset.synthetic_doc_cvb import CVBSynDoc
from scipy.special import gammaln
import eval.eval_tools as eval_tools
from itertools import chain, combinations
import time

class TopicTrackingVIModel(object):

    def __init__(self, beta, data):
        self.beta = beta
        self.W = data.W
        self.data = data
        
        #self.dp_matrices = self.init_dp_matrices()
        self.dp_matrix = np.zeros((self.data.max_doc_len, self.data.max_doc_len))
        self.best_lm_docs = self.init_best_lm_docs()#keeps track which (other) documents' sentences contributed to the highest likelihood of a language model
        self.best_lm_word_counts = self.init_best_lm_word_counts()#best_lm_word_counts are the word count from OTHER documents that allowed the highest likelihood of a language model
        self.best_seg_tracker = self.init_seg_tracker()
        self.doc_combs_list = self.init_doc_combs()#All n possible combinations (up to the number of documents). Its a list of pairs where the first element is the combination and second the remaining docs
        self.best_segmentation = [[] for i in range( self.data.max_doc_len)]
        
        self.seg_ll_C = gammaln(self.beta.sum())-gammaln(self.beta).sum()
    
    def init_dp_matrices(self):
        dp_matrices = []
        for doc_i in range(self.data.n_docs):
            doc_len = self.data.doc_len(doc_i)
            doc_i_dp_matrix = np.zeros((doc_len, doc_len))
            dp_matrices.append(doc_i_dp_matrix)
        return dp_matrices
    
    def init_cum_sums(self):
        cum_sums = []
        for doc_i in range(self.data.n_docs):
            cum_sums.append(np.cumsum(self.data.doc_word_counts(doc_i), axis=0))
        return cum_sums
        
    def init_best_lm_docs(self):
        best_lm_docs = []
        for doc_i in range(self.data.n_docs):
            best_lm_doc_i = []
            for lm in range(self.data.doc_len(doc_i)):
                lm_doc_i = []
                for doc_j in range(self.data.n_docs):
                    lm_doc_i.append(-1)
                best_lm_doc_i.append(lm_doc_i)
            best_lm_docs.append(best_lm_doc_i)
        return best_lm_docs
        
    def init_best_lm_word_counts(self):
        best_lm_word_counts = [[] for i in range(self.data.n_docs)]
        for doc_lm_word_counts in best_lm_word_counts:
            for lm in range(self.data.max_doc_len):
                word_counts_dict = {"ll": -np.inf, "wc": np.zeros(self.W)}
                doc_lm_word_counts.append(word_counts_dict)
        return best_lm_word_counts
    
    def init_seg_tracker(self):
        best_seg_tracker = []
        for doc_i in range(self.data.n_docs):
            best_seg_tracker.append(np.zeros(self.data.doc_len(doc_i), dtype=np.int32))
        
        return best_seg_tracker
    
    def init_doc_combs(self):
        all_docs = set(range(self.data.n_docs))
        doc_combs = set(chain.from_iterable(combinations(all_docs, r) for r in range(1,len(all_docs)+1)))
        doc_combs_list = []
        for doc_comb in doc_combs:
            other_docs = all_docs - doc_comb
            doc_combs_list.append([doc_comb, other_docs])
        return doc_combs_list
        
    def slice_docs(self, u, doc_i):
        '''
        Computes the commulative counts of u sentence from all documents
        except doc_i.
        :param u: sentence index
        :param doc_i: document index
        '''
        word_counts = np.zeros(self.W)
        for d in range(self.data.n_docs):
            if d == doc_i:
                '''
                We just want to compute the counts from other documents,
                we need them later in the update part.
                '''
                continue
            
            if self.data.doc_len(doc_i) >= u:
                word_counts += self.data.doc_word_counts(d)[u]
        return word_counts
    
    def possible_other_docs(self, doc, u, lm):
        '''
        Returns the documents (different from doc) that are consistent
        with previous segmentation points. For example, if u=3 and lm=4
        we can only consider other documents for which their u=2 was used
        in lm=4.
        :param doc: document index
        :param u: sentence index
        :param lm: language model index
        '''
        if u == lm:
            '''
            This is the diagonal case for which there are no sentences from other docs
            Thus, there are nos restrictions of other docs.
            '''
            possible_docs = list(range(self.data.n_docs))
            possible_docs.pop(doc)
            return possible_docs
        
        possible_docs = []
        lm_docs = self.best_lm_docs[doc][lm]
        for doc_i in range(self.data.n_docs):
            if doc_i == doc:
                continue
            u_prev = lm_docs[doc_i]
            if u_prev == u-1:
                possible_docs.append(doc_i)
        return possible_docs
        
    def aggregate_u_counts(self, u, docs):
        '''
        Computes the commulative counts of u sentence from all documents.
        :param u: sentence index
        :param doc_i: list of documents
        '''
        word_counts = np.zeros(self.W)
        for doc_i in docs:
            if self.data.doc_len(doc_i) >= u:
                word_counts += self.data.doc_word_counts(doc_i)[u]
        return word_counts
    
    def update_best_lm_docs(self, doc, u, lm, best_doc_comb):
        for doc_i in best_doc_comb:
            self.best_lm_docs[doc][lm][doc_i] = u
    
    def update_best_lm_word_counts(self, doc_i, lm, word_counts, seg_ll):
        '''
        Checks if the current lm estimate was higher than before (when using one
        less sentence). If yes, the tracker for the word counts is updated.
        :param doc_i: index of document
        :param lm: language model
        :param word_counts: word counts from other documents from the current lm estimate
        :param seg_ll: segment log likelihood from the current lm estimate
        '''
        if seg_ll > self.best_lm_word_counts[doc_i][lm]["ll"]:
            self.best_lm_word_counts[doc_i][lm]["wc"] = word_counts
            self.best_lm_word_counts[doc_i][lm]["ll"] = seg_ll
    
    def get_prev_seg_ll(self, u, lm, doc):
        '''
        Returns the log likelihood of the best segmentation
        of a document without sentence u.  #TODO: rebuild this behavior!!!!
        :param u: sentence
        :param lm: language model (column in the DP matrix)
        :param doc: document
        '''
        if u == 0 or lm == 0:
            #We are just in the first sentence or language model, no previous seg exists
            return 0.0
        
        best_seg_point = self.best_seg_tracker[doc][lm-1]
        prev_seg_ll = self.dp_matrices[doc][lm-1][best_seg_point]
        return prev_seg_ll
    
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
    
    def docs_segment_word_counts(self, u_begin, u_end, docs):
        '''
        Returns the commulative counts of the documents from their
        segment u_begin to u_end.
        :param u_begin: segment beginning sentence index
        :param u_end: segment end sentence index
        :param docs: list of documents
        '''
        word_counts = np.zeros(self.W)
        for doc_i in docs:
            if self.data.doc_len(doc_i) >= u_end:
                u_end_real = self.data.doc_len(doc_i)
            else:
                u_end_real = u_end
            word_counts += self.data.doc_word_counts(doc_i)[u_begin:u_end_real+1]
        return word_counts
        
    def fit_sentence(self, u, docs, u_clusters):
        '''
        Adds the u sentece of the docs to the best u_cluster. That is,
        the cluster where u-1 (of the corresponding doc) is located. Note
        that the cluster is for the best likelihood, thus, we need to add
        u to the same cluster as u-1.
        :param u: sentence index
        :param docs: list of documents
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        for doc in docs:
            for u_cluster in reversed(u_clusters):#We reverse because we are only checking the document list of the cluster
                if u_cluster.has_doc(u-1, doc):
                    u_cluster.add_sent(u,doc)
                    break
        return u_clusters
    
    def segment_u(self, u, lm):
        '''
        Estimates, for all documents, the best segmentation
        from u to lm (column index in the DP matrix).
        :param u: sentence index
        :param lm: language model index
        '''
        if lm == 0:#The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            word_counts = self.docs_segment_word_counts(u, lm, range(self.data.n_docs))
            segmentation_ll = self.segment_ll(word_counts)
            u_clusters = [SentenceCluster(u, lm, list(range(self.data.n_docs)), self.data)]
            return segmentation_ll, u_clusters
            
        best_seg_ll = -np.inf
        best_seg_clusters = None
        for doc_comb, other_docs in self.doc_combs_list:
            best_prev_seg = copy.deepcopy(self.best_segmentation[lm-1])
            doc_comb_seg = SentenceCluster(u, lm, doc_comb, self.data)
            updated_clusters = self.fit_sentence(u, other_docs, best_prev_seg)
            segmentation_ll = 0.0
            for u_cluster in updated_clusters:
                segmentation_ll += self.segment_ll(u_cluster.get_word_counts())
            segmentation_ll = self.segment_ll(doc_comb_seg.word_counts)
            if segmentation_ll >= best_seg_ll:
                best_seg_ll = segmentation_ll
                best_prev_seg.append(doc_comb_seg)
                best_seg_clusters = best_prev_seg
        return best_seg_ll, best_seg_clusters
            
    def dp_segmentation(self):
        for u in range(self.data.max_doc_len):
            best_seg_ll = -np.inf
            best_seg_clusters = None
            for lm in range(u+1):
                #print("u: %d lm: %d"%(u, lm))
                seg_ll, seg_clusters = self.segment_u(u, lm)
                if seg_ll > best_seg_ll:
                    best_seg_ll = seg_ll
                    best_seg_clusters = seg_clusters
            self.best_segmentation[lm] = best_seg_clusters
                
    def get_segmentation(self, doc_i):#TODO: needs to be redone, segmentation is now based on the SentenceCluster class
        '''
        Returns the final segmentation for a document.
        This is done by backtracking the best segmentations
        in a bottom-up fashion.
        :param doc_i: document index
        '''
        u = len(self.best_seg_tracker[doc_i])-1
        seg_points = []
        while 1:
            seg_point = self.best_seg_tracker[doc_i][u]-1
            if seg_point == -1:
                break
            seg_points.append(seg_point)
            u = seg_point-1
            if u == -1:
                break
        seg_points.reverse()
        hyp_seg = []
        seg_begin = 0
        for seg_point in seg_points:
            seg_len = seg_point-seg_begin
            hyp_seg += [0]*(seg_len)+[1]
            seg_begin = seg_point+1
            
        seg_len = self.data.doc_len(doc_i)-seg_begin
        hyp_seg += [0]*seg_len
        hyp_seg[-1] = 0
        return hyp_seg
    
    def get_all_segmentations(self):
        '''
        Returns a single vector with the final
        segmentation for all documents.
        '''
        all_segs = []
        for doc_i in range(self.data.n_docs):
            all_segs += self.get_segmentation(doc_i)
        return all_segs
    
class Data(object):
    '''
    Wrapper class for MultiDocument object. Represent the full collection of documents.
    In this segmentor implementation it is convenient to have
    individual word counts for each document. 
    '''
    def __init__(self, docs):
        self.W = docs.W
        self.n_docs = docs.n_docs
        self.doc_lens = []
        self.docs_word_counts = []
        self.multi_doc_slicer(docs)
        self.max_doc_len = np.max(self.doc_lens)
        
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
    def __init__(self, u, lm, docs, data):
        self.data = data
        self.u_list = []
        self.doc_list = []
        self.word_counts = np.zeros(self.data.W)
        seg_len = u-lm+1
        for doc_i in docs:
            if self.data.doc_len(doc_i) >= u:
                u_end = self.data.doc_len(doc_i)
            else:
                u_end = u_end
            self.u_list += list(range(lm, u_end+1))
            self.doc_list += [doc_i]*seg_len
            self.word_counts += self.data.doc_word_counts(doc_i)[lm:u_end+1]
    
    def has_doc(self, doc_i):
        return doc_i in self.doc_list
    
    def add_sent(self, u, doc_i):
        self.u_list.append(u)
        self.doc_list.append(doc_i)
        self.word_counts += self.data.doc_word_counts(doc_i)[u]
        
    def get_word_counts(self):
        return self.word_counts
            
def sigle_vs_md_eval(doc_synth, beta):
    '''
    Print the WD results when segmenting single documents
    and all of them simultaneously (multi-doc model)
    :param doc_synth: collection of synthetic documents
    :param beta: beta prior vector
    '''
    single_docs = doc_synth.get_single_docs()
    single_doc_wd = []
    start = time.time()
    for doc in single_docs:
        data = Data(doc)
        vi_tt_model = TopicTrackingVIModel(beta, data)
        vi_tt_model.dp_segmentation()
        single_doc_wd += eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc)
    end = time.time()
    sd_time = (end - start)
        
    single_doc_wd = ['%.3f' % wd for wd in single_doc_wd]
    data = Data(doc_synth)
    vi_tt_model = TopicTrackingVIModel(beta, data)
    start = time.time()
    vi_tt_model.dp_segmentation()
    end = time.time()
    md_time = (end - start)
    multi_doc_wd = eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth)
    multi_doc_wd = ['%.3f' % wd for wd in multi_doc_wd]
    print("Single:%s time: %f\nMulti: %s time: %f" % (str(single_doc_wd), sd_time, str(multi_doc_wd), md_time))
    
    
W = 10
beta = np.array([0.6]*W)
n_docs = 5
doc_len = 40 
pi = .2
sent_len = 5
doc_synth = CVBSynDoc(beta, pi, sent_len, doc_len, n_docs)
data = Data(doc_synth)

#sigle_vs_md_eval(doc_synth, beta)

vi_tt_model = TopicTrackingVIModel(beta, data)
vi_tt_model.dp_segmentation()
print(eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth))
